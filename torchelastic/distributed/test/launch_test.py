#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from unittest import mock
from unittest.mock import Mock, patch

import torchelastic.distributed.launch as launch
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
from torch.multiprocessing import start_processes
from torchelastic.rendezvous.etcd_server import EtcdServer
from torchelastic.test.test_utils import is_tsan


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


def get_child_pids(pid):
    pgrep = subprocess.Popen(args=f"pgrep -P {pid}", shell=True, stdout=subprocess.PIPE)
    pgrep.wait()
    out = pgrep.stdout.read().decode("utf-8").rstrip().split("\n")
    pids = []
    for pid in out:
        if pid:
            pids.append(int(pid))
    return pids


def pid_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class MockException(Exception):
    pass


class LaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()
        cls._etcd_endpoint = cls._etcd_server.get_endpoint()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # remove any lingering environment variables
        for env in os.environ.keys():
            if env.startswith("PET_"):
                del os.environ[env]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_launch_user_script_python(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            "--monitor_interval=1",
            "--start_method=fork",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_launch_user_script_bash(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node

        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            "--monitor_interval=1",
            "--start_method=fork",
            "--no_python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no_python cannot be used with --module
            launch.main(args + ["--module"] + script_args)

        launch.main(args + script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_launch_with_env_vars(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node

        os.environ["PET_NNODES"] = str(nnodes)
        os.environ["PET_NPROC_PER_NODE"] = str(nproc_per_node)
        os.environ["PET_RDZV_BACKEND"] = "etcd"
        os.environ["PET_RDZV_ENDPOINT"] = self._etcd_endpoint
        os.environ["PET_RDZV_ID"] = run_id
        os.environ["PET_MONITOR_INTERVAL"] = "1"
        os.environ["PET_START_METHOD"] = "fork"
        os.environ["PET_NO_PYTHON"] = "1"

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        with self.assertRaises(ValueError):
            # --no_python cannot be used with --module
            os.environ["PET_MODULE"] = "1"
            launch.main(script_args)

        os.environ["PET_MODULE"] = "0"
        launch.main(script_args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_wrapper_fn_kill_script_process(self):
        """
        tests that the wrapper_fn properly terminates
        the script process (the script process is the subprocess of
        the agent)
        """
        nprocs = 2
        sleep = 300

        # wraps wrapper_fn to be torch.multiprocessing compatible
        # which requires rank to be passed as first arugment
        def wrap_wrap(rank, *args):
            launch.wrapper_fn(*args)

        context = start_processes(
            fn=wrap_wrap,
            args=(None, (path("bin/sleep_script.py"), "--sleep", f"{sleep}")),
            nprocs=nprocs,
            join=False,
            start_method="fork",
        )
        # quick check to see that the wrapper_fn started running
        # without this join() call we don't see an exception on typos
        # and other silly mistakes (silently fails)
        context.join(timeout=-1)

        script_pids = []
        for wrapper_fn_pid in context.pids():
            script_pid = get_child_pids(wrapper_fn_pid)
            # there should only be one child of wrapper_fn
            self.assertEqual(1, len(script_pid))
            script_pids.append(script_pid[0])

        for wrapper_fn_proc in context.processes:
            wrapper_fn_proc.terminate()
            wrapper_fn_proc.join()

        for script_pid in script_pids:
            self.assertFalse(pid_exists(script_pid))

    def _test_nproc_launch_configuration(self, nproc_type, expected_number):
        run_id = str(uuid.uuid4().int)
        nnodes = 1

        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_type}",
            "--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            "--monitor_interval=1",
            "--start_method=fork",
            "--no_python",
        ]

        script_args = [path("bin/test_script.sh"), f"{self.test_dir}"]

        launch.main(args + script_args)

        world_size = nnodes * expected_number
        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_nproc_launch_auto_configurations(self):
        self._test_nproc_launch_configuration("auto", os.cpu_count())

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_nproc_launch_number_configurations(self):
        self._test_nproc_launch_configuration("4", 4)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_nproc_launch_unknown_configurations(self):
        with self.assertRaises(ValueError):
            self._test_nproc_launch_configuration("unknown", 4)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=3)
    def test_nproc_gpu_launch_configurations(self, _mock1, _mock2):
        self._test_nproc_launch_configuration("auto", 3)
        self._test_nproc_launch_configuration("gpu", 3)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_launch_elastic(self):
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        # we are only launching 1 node (even though max = 2)
        world_size = nproc_per_node
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            "--monitor_interval=1",
            "--start_method=fork",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_launch_elastic_worker_raise_exception(self):
        """
        Asserts that when the worker program fails the launcher catches
        WorkerGroupFailureException and re-raises the min_rank's inner exception,
        which will be a regular ``Exception`` with an message that looks like:

        ::
         -- Process 0 terminated with the following error:
         Traceback (most recent call last):
             File "...<omitted>..., line ## in _foo
                fn(i, args...)
             File "...<omitted>..., line ## in bar
                ...<omitted>...

        """
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            "--monitor_interval=1",
            "--max_restarts=0",
            "--start_method=fork",
            path("bin/test_script.py"),
            "--fail",
        ]

        with self.assertRaisesRegex(
            Exception, "-- Process 0 terminated with the following error:"
        ):
            launch.main(args)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    @mock.patch("torchelastic.agent.server.local_elastic_agent.LocalElasticAgent.run")
    def test_launch_elastic_agent_raise_exception(self, mock_agent_run):
        """
        Asserts that when the agent raises an exception
        the launcher re-raises the original exception
        """
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            "--monitor_interval=1",
            "--max_restarts=0",
            "--start_method=fork",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]

        mock_agent_run.side_effect = MockException
        with self.assertRaises(MockException):
            launch.main(args)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_launch_standalone(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--standalone",
            "--monitor_interval=1",
            "--start_method=fork",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_launch_elastic_multiple_agents(self):
        run_id = str(uuid.uuid4().int)
        min_nodes = 1
        max_nodes = 2
        nproc_per_node = 4
        nnodes = 2
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={min_nodes}:{max_nodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            "--monitor_interval=1",
            "--start_method=fork",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]
        procs = []
        for _ in range(nnodes - 1):
            p = mp.Process(target=launch.main, args=[args])
            procs.append(p)
            p.start()
        launch.main(args)
        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )

    def test_min_max_nodes_parse(self):
        min_nodes, max_nodes = launch.parse_min_max_nnodes("1")
        self.assertTrue(min_nodes, max_nodes)
        self.assertTrue(1, min_nodes)
        min_nodes, max_nodes = launch.parse_min_max_nnodes("2:20")
        self.assertTrue(2, min_nodes)
        self.assertTrue(20, max_nodes)
        with self.assertRaises(RuntimeError):
            launch.parse_min_max_nnodes("2:20:30")

    @patch("torchelastic.distributed.launch.LocalElasticAgent")
    def test_launch_rdzv_shutdown(self, _):
        nnodes = 1
        nproc_per_node = 4
        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            "--monitor_interval=1",
            "--start_method=fork",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]
        rdzv_handler_mock = Mock()
        with patch(
            "torchelastic.rendezvous.registry.get_rendezvous_handler"
        ) as param_mock:
            param_mock.return_value = rdzv_handler_mock
            launch.main(args)
            rdzv_handler_mock.shutdown.assert_called_once()
