#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import multiprocessing
import os
import time
import unittest
import uuid

import torch
import torch.distributed as dist
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
from test_utils import is_tsan
from torchelastic.agent.server.api import (
    WorkerGroupFailureException,
    WorkerSpec,
    WorkerState,
)
from torchelastic.agent.server.local_elastic_agent import LocalElasticAgent
from torchelastic.rendezvous.etcd_server import EtcdServer


def _happy_function():
    return


def _sad_function():
    raise RuntimeError("sad because i throw")


def _bipolar_function():
    rank = int(os.environ["RANK"])
    if rank % 2 == 0:
        _happy_function()
    else:
        _sad_function()


def _distributed_sum(wait):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    t = torch.tensor(rank)

    time.sleep(wait)
    dist.all_reduce(t, op=dist.reduce_op.SUM)

    expected = sum(range(world_size))
    actual = t.item()
    if expected != actual:
        raise RuntimeError(f"Expected rank sum {expected}, got {actual}")


def _return_rank_times(a):
    return int(os.environ["RANK"]) * a


def _check_env_function():
    # just check these env vars exist, os.environ[...] will naturally throw
    # if the variable does not exist
    os.environ["RANK"]
    os.environ["LOCAL_RANK"]
    os.environ["GROUP_RANK"]
    os.environ["LOCAL_WORLD_SIZE"]
    os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"]
    os.environ["MASTER_PORT"]
    os.environ["TORCHELASTIC_RESTART_COUNT"]
    os.environ["TORCHELASTIC_MAX_RESTARTS"]


def _run_agent(run_id, etcd_host, etcd_port, min_size, max_size, wait=0):
    rdzv_handler = dist.rendezvous(
        f"etcd://{etcd_host}:{etcd_port}/{run_id}"
        f"?min_workers={min_size}"
        f"&max_workers={max_size}"
    )
    spec = WorkerSpec(
        role="test_trainer",
        local_world_size=8,
        fn=_distributed_sum,
        args=(wait,),
        rdzv_handler=rdzv_handler,
        max_restarts=2,
        monitor_interval=1,
    )

    agent = LocalElasticAgent(spec, start_method="fork")
    agent.run()


class LocalElasticAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def _get_worker_spec(
        self, fn, args=(), max_restarts=1, num_agents=1, monitor_interval=0.1
    ):
        run_id = str(uuid.uuid4().int)
        rdzv_handler = dist.rendezvous(
            f"etcd://{self._etcd_server.get_endpoint()}/{run_id}"
            f"?min_workers={num_agents}"
            f"&max_workers={num_agents}"
        )
        spec = WorkerSpec(
            role="test_trainer",
            local_world_size=8,
            fn=fn,
            args=args,
            rdzv_handler=rdzv_handler,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
        )
        return spec

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_happy_function(self):
        spec = self._get_worker_spec(fn=_happy_function)
        agent = LocalElasticAgent(spec, start_method="fork")
        agent.run()

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_distributed_sum(self):
        spec = self._get_worker_spec(fn=_distributed_sum, args=(0,))
        agent = LocalElasticAgent(spec, start_method="fork")
        agent.run()

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_sad_function(self):
        spec = self._get_worker_spec(fn=_sad_function, max_restarts=2)
        agent = LocalElasticAgent(spec, start_method="fork")
        with self.assertRaises(WorkerGroupFailureException) as cm:
            agent.run()

        excs = cm.exception.get_worker_exceptions()
        for i in range(spec.local_world_size):
            self.assertTrue(isinstance(excs[i], Exception))

        self.assertEqual(WorkerState.FAILED, agent.get_worker_group().state)
        self.assertEqual(0, agent._remaining_restarts)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_bipolar_function(self):
        spec = self._get_worker_spec(fn=_bipolar_function, max_restarts=2)
        agent = LocalElasticAgent(spec, start_method="fork")
        with self.assertRaises(Exception):
            agent.run()
        self.assertEqual(WorkerState.FAILED, agent.get_worker_group().state)
        self.assertEqual(0, agent._remaining_restarts)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_check_env_function(self):
        spec = self._get_worker_spec(fn=_check_env_function, max_restarts=2)
        agent = LocalElasticAgent(spec, start_method="fork")
        agent.run()

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_get_worker_return_values(self):
        spec = self._get_worker_spec(fn=_return_rank_times, args=(2,))
        agent = LocalElasticAgent(spec, start_method="fork")
        ret_vals = agent.run()

        self.assertEqual(spec.local_world_size, len(ret_vals))
        for i in range(spec.local_world_size):
            self.assertEqual(i * 2, ret_vals[i])

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_double_agent_happy(self):
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        nnodes = 2
        run_id = str(uuid.uuid4().int)

        procs = []
        for _ in range(nnodes - 1):
            p = multiprocessing.Process(
                target=_run_agent, args=(run_id, host, port, nnodes, nnodes)
            )
            procs.append(p)
            p.start()

        # run one on the main process for debugging
        _run_agent(run_id, host, port, nnodes, nnodes)

        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_double_agent_fault_tolerance(self):
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        nnodes = 2
        run_id = str(uuid.uuid4().int)

        procs = []
        for _ in range(nnodes):
            p = multiprocessing.Process(
                target=_run_agent, args=(run_id, host, port, nnodes, nnodes)
            )
            procs.append(p)
            p.start()

        # restart odd agents
        for i in range(nnodes):
            if i % 2 != 0:
                procs[i].kill()
                p = multiprocessing.Process(
                    target=_run_agent, args=(run_id, host, port, nnodes, nnodes)
                )
                procs[i] = p
                p.start()

        for i in range(nnodes):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_double_agent_elastic(self):
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        min_size = 1
        max_size = 2
        run_id = str(uuid.uuid4().int)

        procs = []
        for _ in range(max_size):
            p = multiprocessing.Process(
                target=_run_agent, args=(run_id, host, port, min_size, max_size)
            )
            procs.append(p)
            p.start()

        # kill odd agents
        for i in range(max_size):
            if i % 2 != 0:
                procs[i].kill()

        for i in range(max_size):
            if i % 2 == 0:
                p = procs[i]
                p.join()
                self.assertEqual(0, p.exitcode)
