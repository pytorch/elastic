#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import uuid
from typing import Any, Dict
from unittest.mock import patch

import torch.distributed as dist
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
from p2p.etcd_server_fixture import EtcdServerFixture
from torchelastic.agent.server.api import (
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
    _get_fq_hostname,
)
from torchelastic.rendezvous import RendezvousHandler


def do_nothing():
    pass


class WorkerStateTest(unittest.TestCase):
    def test_is_running(self):
        for state in WorkerState:
            if state == WorkerState.HEALTHY or state == WorkerState.UNHEALTHY:
                self.assertTrue(WorkerState.is_running(state))
            else:
                self.assertFalse(WorkerState.is_running(state))


class WorkerGroupTest(unittest.TestCase):
    def test_worker_group_constructor(self):
        spec = WorkerSpec(
            role="test_trainer",
            local_world_size=4,
            fn=do_nothing(),
            args=(),
            rdzv_handler=None,
            max_restarts=50,
            monitor_interval=1,
        )
        worker_group = WorkerGroup(spec)

        self.assertEqual(WorkerState.INIT, worker_group.state)

        workers = worker_group.workers
        self.assertEqual(4, len(workers))

        # validate full, consecutive local ranks
        self.assertSetEqual(set(range(4)), {w.local_rank for w in workers})

        # global_rank, world_size are assigned after rdzv
        # id is assigned after starting worker (by the agent)
        # validate there are None
        for w in workers:
            self.assertIsNone(w.global_rank)
            self.assertIsNone(w.world_size)
            self.assertIsNone(w.id)

        # rank and store are assigned after rdzv; validate that they are None
        self.assertIsNone(worker_group.group_rank)
        self.assertIsNone(worker_group.store)


class TestAgent(SimpleElasticAgent):
    def __init__(self, spec):
        super().__init__(spec)
        self.stop_workers_call_count = 0
        self.start_workers_call_count = 0

    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        # workers are fake, nothing to stop; just clear the rdzv info
        worker_group.group_rank = None
        worker_group.group_world_size = None
        for w in worker_group.workers:
            w.id = None
            w.global_rank = None
            w.world_size = None
        self.stop_workers_call_count += 1

    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        # crate fake workers; make worker id equal to global rank
        ids = {}
        for worker in worker_group.workers:
            ids[worker.local_rank] = worker.global_rank
        self.start_workers_call_count += 1
        return ids

    def _monitor_workers(self, worker_group: WorkerGroup) -> WorkerState:
        raise NotImplementedError("mock this method")


class SimpleElasticAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServerFixture()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def _get_worker_spec(self, max_restarts=1, monitor_interval=1.0):
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        run_id = str(uuid.uuid4().int)
        rdzv_handler = dist.rendezvous(
            f"etcd://{host}:{port}/{run_id}?min_workers=1&max_workers=1"
        )
        spec = WorkerSpec(
            role="test_trainer",
            local_world_size=8,
            fn=do_nothing,
            args=(),
            rdzv_handler=rdzv_handler,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
        )
        return spec

    def test_agent_constructor(self):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        self.assertEquals(WorkerState.INIT, worker_group.state)
        self.assertEquals(spec.max_restarts, agent._remaining_restarts)

    def test_rendezvous(self):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        agent._rendezvous(worker_group)

        # single agent rdzv
        self.assertEqual(1, worker_group.group_world_size)
        self.assertEqual(0, worker_group.group_rank)

        master_addr, master_port = agent._get_master_addr_port(worker_group.store)

        self.assertEqual(_get_fq_hostname(), master_addr)
        self.assertTrue(master_port > 0)

        rank_set = {w.global_rank for w in worker_group.workers}
        for w in worker_group.workers:
            self.assertIsNone(w.id)
            local_world_size = spec.local_world_size
            group_world_size = worker_group.group_world_size
            group_rank = worker_group.group_rank

            self.assertEqual(local_world_size * group_world_size, w.world_size)
            self.assertEqual(
                local_world_size * group_rank + w.local_rank, w.global_rank
            )
            self.assertSetEqual(set(range(w.world_size)), rank_set)

    def test_initialize_workers(self):
        spec = self._get_worker_spec(max_restarts=1)
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()
        agent._initialize_workers(worker_group)

        self.assertEqual(WorkerState.HEALTHY, worker_group.state)
        for i in range(spec.local_world_size):
            worker = worker_group.workers[i]
            self.assertEqual(worker.id, worker.global_rank)

    def test_restart_workers(self):
        spec = self._get_worker_spec()
        agent = TestAgent(spec)
        worker_group = agent.get_worker_group()

        num_restarts = 3
        for _ in range(0, num_restarts):
            agent._restart_workers(worker_group)
            self.assertEqual(WorkerState.HEALTHY, worker_group.state)

            # test_rendezvous and test_initialize_workers
            # already validates the correctness of these fields
            # simply validate that they are not None
            # (e.g. that they get assigned)
            self.assertIsNotNone(worker_group.group_rank)
            self.assertIsNotNone(worker_group.group_world_size)
            for w in worker_group.workers:
                self.assertIsNotNone(w.id)
                self.assertIsNotNone(w.global_rank)
                self.assertIsNotNone(w.world_size)

        self.assertEqual(num_restarts, agent.start_workers_call_count)
        self.assertEqual(num_restarts, agent.stop_workers_call_count)

    @patch.object(
        TestAgent,
        "_monitor_workers",
        side_effect=[WorkerState.HEALTHY, WorkerState.HEALTHY, WorkerState.SUCCEEDED],
    )
    def test_run_happy_path(self, mock_monitor_workers):
        # worker starts
        # is always healthy
        # then succeeds
        max_restarts = 10
        spec = self._get_worker_spec(max_restarts)
        agent = TestAgent(spec)

        agent.run()

        # no failure, no membership changes -> no retries
        self.assertEquals(max_restarts, agent._remaining_restarts)

    @patch.object(TestAgent, "_initialize_workers", side_effect=RuntimeError())
    def test_run_initialization_failure(self, mock_initialize_workers):
        spec = self._get_worker_spec()
        agent = TestAgent(spec)
        worker_group = agent._worker_group

        with self.assertRaises(RuntimeError):
            agent.run()

        self.assertEqual(WorkerState.INIT, worker_group.state)

    def test_run_max_retries_exceeded(self):
        for restartable_state in [WorkerState.FAILED, WorkerState.UNHEALTHY]:
            with patch.object(
                TestAgent, "_monitor_workers", return_value=restartable_state
            ) as mock_monitor_workers:
                spec = self._get_worker_spec(max_restarts=3, monitor_interval=0.1)
                agent = TestAgent(spec)
                worker_group = agent._worker_group

                with self.assertRaises(Exception):
                    agent.run()

                self.assertEqual(WorkerState.FAILED, worker_group.state)
                self.assertEqual(0, agent._remaining_restarts)
                # one monitor call for each retry + one to monitor the last retry
                self.assertEqual(spec.max_restarts + 1, mock_monitor_workers.call_count)

    @patch.object(
        TestAgent,
        "_monitor_workers",
        side_effect=[
            WorkerState.HEALTHY,
            WorkerState.HEALTHY,
            WorkerState.HEALTHY,
            WorkerState.SUCCEEDED,
        ],
    )
    @patch.object(RendezvousHandler, "num_nodes_waiting", side_effect=[1, 1, 0])
    def test_run_membership_change(self, mock_monitor_workers, mock_num_nodes_waiting):
        spec = self._get_worker_spec(max_restarts=1, monitor_interval=0.1)
        agent = TestAgent(spec)
        worker_group = agent._worker_group

        agent.run()
        self.assertEquals(WorkerState.SUCCEEDED, worker_group.state)

    @patch.object(TestAgent, "_monitor_workers", return_value=WorkerState.UNKNOWN)
    def test_run_unknown_state(self, mock_monitor_workers):
        # when the state is unknown we exit immediately; no retries
        spec = self._get_worker_spec(max_restarts=100, monitor_interval=0.1)
        agent = TestAgent(spec)
        worker_group = agent._worker_group

        with self.assertRaises(Exception):
            agent.run()

        self.assertEqual(WorkerState.UNKNOWN, worker_group.state)
        self.assertEqual(1, mock_monitor_workers.call_count)
        self.assertEqual(spec.max_restarts, agent._remaining_restarts)
