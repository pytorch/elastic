#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import tempfile
import unittest
import uuid

import torchelastic.distributed.launch as launch
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
from test_utils import is_tsan
from torchelastic.rendezvous.etcd_server import EtcdServer


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


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

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_launch_user_script_python(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            f"--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            f"--monitor_interval=1",
            f"--start_method=fork",
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
    def test_launch_user_script_bash(self):
        run_id = str(uuid.uuid4().int)
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node

        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            f"--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            f"--monitor_interval=1",
            f"--start_method=fork",
            f"--no_python",
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
            f"--rdzv_backend=etcd",
            f"--rdzv_endpoint={self._etcd_endpoint}",
            f"--rdzv_id={run_id}",
            f"--monitor_interval=1",
            f"--start_method=fork",
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
    def test_launch_with_etcd(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        args = [
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc_per_node}",
            f"--with_etcd",
            f"--monitor_interval=1",
            f"--start_method=fork",
            path("bin/test_script.py"),
            f"--touch_file_dir={self.test_dir}",
        ]
        launch.main(args)

        # make sure all the workers ran
        # each worker touches a file with its global rank as the name
        self.assertSetEqual(
            {str(i) for i in range(world_size)}, set(os.listdir(self.test_dir))
        )
