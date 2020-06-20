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
import torch.distributed.rpc as rpc
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
from test_utils import is_tsan
from torch.distributed.rpc.backend_registry import BackendType
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

    expected_sum = sum(range(world_size))
    actual = t.item()
    if expected_sum != actual:
        raise RuntimeError(f"Expected rank sum {expected_sum}, got {actual}")


def _simulate_work(wait):
    time.sleep(wait)
    rank = int(os.environ["RANK"])
    return rank


def _check_rank_assignment():
    group_rank = int(os.environ["GROUP_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    role_rank = int(os.environ["ROLE_RANK"])
    role_world_size = int(os.environ["ROLE_WORLD_SIZE"])
    return (group_rank, rank, world_size, role_rank, role_world_size)


def echo(msg):
    return msg


def _return_rank_times(a):
    return int(os.environ["RANK"]) * a


def _check_env_function():
    # just check these env vars exist, os.environ[...] will naturally throw
    # if the variable does not exist
    os.environ["RANK"]
    os.environ["LOCAL_RANK"]
    os.environ["ROLE_RANK"]
    os.environ["GROUP_RANK"]
    os.environ["LOCAL_WORLD_SIZE"]
    os.environ["ROLE_WORLD_SIZE"]
    os.environ["WORLD_SIZE"]
    os.environ["MASTER_ADDR"]
    os.environ["MASTER_PORT"]
    os.environ["TORCHELASTIC_RESTART_COUNT"]
    os.environ["TORCHELASTIC_MAX_RESTARTS"]
    os.environ["TORCHELASTIC_RUN_ID"]


def _run_agent(
    run_id,
    etcd_host,
    etcd_port,
    min_size,
    max_size,
    func_to_run,
    args,
    local_world_size=8,
    role="test_trainer",
    output_dict=None,
    agent_barrier_timeout=300,
):
    rdzv_handler = dist.rendezvous(
        f"etcd://{etcd_host}:{etcd_port}/{run_id}"
        f"?min_workers={min_size}"
        f"&max_workers={max_size}"
    )
    spec = WorkerSpec(
        role=role,
        local_world_size=local_world_size,
        fn=func_to_run,
        args=args,
        rdzv_handler=rdzv_handler,
        max_restarts=2,
        monitor_interval=1,
    )

    agent = LocalElasticAgent(
        spec, start_method="fork", exit_barrier_timeout=agent_barrier_timeout
    )
    res = agent.run()
    if output_dict is not None:
        key = str(uuid.uuid4().int)
        output_dict[key] = (role, res)


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

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_happy_function(self):
        spec = self._get_worker_spec(fn=_happy_function)
        agent = LocalElasticAgent(spec, start_method="fork")
        agent.run()

    def _get_worker_spec(
        self,
        fn,
        args=(),
        max_restarts=1,
        num_agents=1,
        monitor_interval=0.1,
        local_world_size=8,
    ):
        run_id = str(uuid.uuid4().int)
        rdzv_handler = dist.rendezvous(
            f"etcd://{self._etcd_server.get_endpoint()}/{run_id}"
            f"?min_workers={num_agents}"
            f"&max_workers={num_agents}"
        )
        spec = WorkerSpec(
            role="test_trainer",
            local_world_size=local_world_size,
            fn=fn,
            args=args,
            rdzv_handler=rdzv_handler,
            max_restarts=max_restarts,
            monitor_interval=monitor_interval,
        )
        return spec

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_distributed_sum(self):
        spec = self._get_worker_spec(fn=_distributed_sum, args=(0,))
        agent = LocalElasticAgent(spec, start_method="fork")
        agent.run()

    class RoleConfig:
        __slots__ = ["role", "workers", "num_agents", "workers_num", "role_size"]

        def __init__(
            self, role: str, workers=None, num_agents: int = 0, workers_num: int = 0
        ):
            self.role = role
            self.workers = workers
            if workers_num != 0 and num_agents != 0:
                self.workers = [workers_num] * num_agents
            self.role_size = sum(self.workers)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_correct_rank_assignment_heterogeneous(self):
        roles_config = [
            self.RoleConfig("trainer", workers=[1, 2, 3, 4]),
            self.RoleConfig("ps", workers=[5, 2]),
            # split configuration to run the last one on the main process
            self.RoleConfig("master", workers=[8]),
        ]
        self.run_configuration(roles_config, 25)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_correct_rank_assignment_homogeneous(self):
        num_workers = 4
        roles_config = [
            self.RoleConfig("trainer", num_agents=4, workers_num=num_workers),
            self.RoleConfig("ps", num_agents=2, workers_num=num_workers),
            # split configuration to run the last one on the main process
            self.RoleConfig("master", num_agents=1, workers_num=num_workers),
        ]
        self.run_configuration(roles_config, 28)

    def run_configuration(self, roles_config, expected_world_size):
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        nnodes = sum(len(cfg.workers) for cfg in roles_config)
        run_id = str(uuid.uuid4().int)

        procs = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        default_args = (run_id, host, port, nnodes, nnodes, _check_rank_assignment, ())
        for ind in range(len(roles_config) - 1):
            config = roles_config[ind]
            for num_workers in config.workers:
                p = multiprocessing.Process(
                    target=_run_agent,
                    args=(*default_args, num_workers, config.role, return_dict),
                )
                procs.append(p)
                p.start()

        # run one on the main process for debugging
        config = roles_config[len(roles_config) - 1]
        _run_agent(*default_args, config.workers[0], config.role, return_dict)

        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)
        role_info_dict = {role_info.role: role_info for role_info in roles_config}
        self.verify_rank_consistency(return_dict, role_info_dict, expected_world_size)

    def verify_rank_consistency(self, return_dict, role_info_dict, expected_world_size):
        role_ranks = {}
        global_ranks = []
        grouped_ranks = {}
        for role, res in return_dict.values():
            for (
                group_rank,
                rank,
                world_size,
                role_rank,
                role_world_size,
            ) in res.values():
                role_info_config = role_info_dict[role]
                self.assertEqual(expected_world_size, world_size)
                self.assertEqual(role_info_config.role_size, role_world_size)
                if group_rank not in grouped_ranks:
                    grouped_ranks[group_rank] = []
                grouped_ranks[group_rank].append((rank, role_rank))
                global_ranks.append(rank)
                if role not in role_ranks:
                    role_ranks[role] = []
                role_ranks[role].append(role_rank)
        global_ranks = sorted(global_ranks)
        self.assertEqual(list(range(0, expected_world_size)), global_ranks)
        for role, role_config_info in role_info_dict.items():
            self.assertEqual(
                list(range(0, role_config_info.role_size)), sorted(role_ranks[role])
            )
        # Make sure that each agent assignes consecutive ranks to workes
        # The first argument is the global_rank and the second argument
        # is role_rank
        for ranks_lst in grouped_ranks.values():
            self.verify_ranks_sequential(ranks_lst, 0)
            self.verify_ranks_sequential(ranks_lst, 1)

    def verify_ranks_sequential(self, ranks_pairs, rank_idx):
        ranks = sorted(rank_pair[rank_idx] for rank_pair in ranks_pairs)
        start_rank, end_rank = ranks[0], ranks[-1]
        self.assertEqual(list(range(start_rank, end_rank + 1)), ranks)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_distributed_sum_heterogenous(self):
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        nnodes = 4
        run_id = str(uuid.uuid4().int)

        procs = []
        default_args = (run_id, host, port, nnodes, nnodes, _distributed_sum, (0,))
        for ind in range(nnodes - 1):
            p = multiprocessing.Process(
                target=_run_agent, args=(*default_args, ind + 1)
            )
            procs.append(p)
            p.start()

        # run one on the main process for debugging
        _run_agent(*default_args, 8)

        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

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
    def test_run_check_run_id(self):
        def return_run_id():
            return os.environ["TORCHELASTIC_RUN_ID"]

        spec = self._get_worker_spec(fn=return_run_id, max_restarts=0)
        agent = LocalElasticAgent(spec, start_method="fork")
        ret = agent.run()

        for i in range(spec.local_world_size):
            self.assertEqual(spec.rdzv_handler.get_run_id(), ret[i])

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
                target=_run_agent,
                args=(run_id, host, port, nnodes, nnodes, _distributed_sum, (0,)),
            )
            procs.append(p)
            p.start()

        # run one on the main process for debugging
        _run_agent(run_id, host, port, nnodes, nnodes, _distributed_sum, (0,))

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
                target=_run_agent,
                args=(run_id, host, port, nnodes, nnodes, _distributed_sum, (0,)),
            )
            procs.append(p)
            p.start()

        # restart odd agents
        for i in range(nnodes):
            if i % 2 != 0:
                procs[i].kill()
                p = multiprocessing.Process(
                    target=_run_agent,
                    args=(run_id, host, port, nnodes, nnodes, _distributed_sum, (0,)),
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
                target=_run_agent,
                args=(run_id, host, port, min_size, max_size, _distributed_sum, (0,)),
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

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_torch_rpc(self):
        """
        Simple torch rpc example with torchelastic.
        Creates two agents (to simulate two node job),
        each agent runs a single worker. worker0 calls an rpc_sync on
        worker1.
        """

        # TODO upstream this to torch.distributed.rpc so that users do not have
        # to redundantly set rank as part of name (e.g. worker0) AND also pass
        # it explicitly as an argument to rpc.init_rpc
        def init_rpc(name_prefix, backend):
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            rpc.init_rpc(
                name=f"{name_prefix}{rank}",
                backend=backend,
                rank=rank,
                world_size=world_size,
            )

        def worker_0(queue, msg):
            init_rpc("worker", BackendType.PROCESS_GROUP)
            ret = rpc.rpc_sync(to="worker1", func=echo, args=(msg,))
            queue.put(ret)
            rpc.shutdown()

        def worker_1():
            init_rpc("worker", BackendType.PROCESS_GROUP)
            rpc.shutdown()

        def run_agent(
            run_id, etcd_host, etcd_port, start_method, worker_fn, worker_args=()
        ):
            rdzv_handler = dist.rendezvous(
                f"etcd://{etcd_host}:{etcd_port}/{run_id}"
                f"?min_workers=2"
                f"&max_workers=2"
            )
            spec = WorkerSpec(
                role="test_trainer",
                local_world_size=1,
                fn=worker_fn,
                args=worker_args,
                rdzv_handler=rdzv_handler,
                max_restarts=3,
                monitor_interval=1,
            )

            agent = LocalElasticAgent(spec, start_method)
            agent.run()

        run_id = str(uuid.uuid4().int)
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        start_method = "fork"
        msg = "hello world"
        mp_queue = multiprocessing.get_context(start_method).Queue()

        agent0 = multiprocessing.Process(
            target=run_agent,
            args=(run_id, host, port, start_method, worker_0, (mp_queue, msg)),
        )
        agent1 = multiprocessing.Process(
            target=run_agent, args=(run_id, host, port, start_method, worker_1, ())
        )

        agent0.start()
        agent1.start()

        agent0.join()
        agent1.join()

        self.assertEqual(msg, mp_queue.get())

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_workers_drift_success(self):

        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        nnodes = 2
        run_id = str(uuid.uuid4().int)

        procs = []
        default_args = (run_id, host, port, nnodes, nnodes, _simulate_work)
        for _ in range(nnodes - 1):
            p = multiprocessing.Process(
                target=_run_agent,
                args=(*default_args, (10,), 2, "test_trainer", {}, 30),
            )
            procs.append(p)
            p.start()

        _run_agent(*default_args, (1,), 2, "test_trainer", {}, 30)

        for i in range(nnodes - 1):
            p = procs[i]
            p.join()
            self.assertEqual(0, p.exitcode)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_workers_drift_fail(self):

        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        nnodes = 2
        run_id = str(uuid.uuid4().int)

        procs = []
        default_args = (run_id, host, port, nnodes, nnodes, _simulate_work)
        for _ in range(nnodes - 1):
            p = multiprocessing.Process(
                target=_run_agent,
                args=(*default_args, (60,), 2, "test_trainer", {}, 10),
            )
            procs.append(p)
            p.start()

        # TODO(aivanou): standardize error between different rendezvous stores
        with self.assertRaises(LookupError):
            _run_agent(*default_args, (1,), 2, "test_trainer", {}, 10)
