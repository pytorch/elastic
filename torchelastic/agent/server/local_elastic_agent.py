#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict

import torch.multiprocessing as mp
from torchelastic.agent.server.api import (
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)


log = logging.getLogger(__name__)


class _DistInfo:
    """
    Container for information required to create a torch process group.
    To be created on the agent's process and passed to the worker sub-process.
    Hence this object needs to be a pure data object with no state and
    preferably only primitive member variables
    """

    __slots__ = [
        "rank",
        "world_size",
        "master_addr",
        "master_port",
        "restart_count",
        "max_restarts",
    ]

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        restart_count: int,
        max_restarts: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.restart_count = restart_count
        self.max_restarts = max_restarts


def _wrap(local_rank, dist_infos, fn, args):
    info = dist_infos[local_rank]
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(info.rank)
    os.environ["WORLD_SIZE"] = str(info.world_size)
    os.environ["MASTER_ADDR"] = info.master_addr
    os.environ["MASTER_PORT"] = str(info.master_port)
    os.environ["TORCHELASTIC_RESTART_COUNT"] = str(info.restart_count)
    os.environ["TORCHELASTIC_MAX_RESTARTS"] = str(info.max_restarts)
    fn(*args)


class LocalElasticAgent(SimpleElasticAgent):
    """
    An implementation of ``ElasticAgent`` that handles host-local workers.
    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.
    """

    def __init__(self, spec: WorkerSpec, start_method="spawn"):
        super().__init__(spec)
        self._start_method = start_method
        self._process_context: mp.ProcessContext = None

    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        for proc in self._process_context.processes:
            if proc.is_alive():
                proc.terminate()

    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        dist_infos: Dict[int, _DistInfo] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            dist_infos[local_rank] = _DistInfo(
                worker.global_rank,
                worker.world_size,
                master_addr,
                master_port,
                restart_count,
                spec.max_restarts,
            )

        self._process_context = mp.start_processes(
            fn=_wrap,
            args=(dist_infos, spec.fn, spec.args),
            nprocs=spec.local_world_size,
            join=False,
            daemon=False,
            start_method=self._start_method,
        )

        return {
            local_rank: pid
            for local_rank, pid in enumerate(self._process_context.pids())
        }

    def _monitor_workers(self, worker_group: WorkerGroup) -> WorkerState:
        role = worker_group.spec.role

        # torch process context join() isn't really a join in the
        # traditional sense, it returns True if all the workers have
        # successfully finished, False if some/all are still running
        # and throws an Exception if some/all of them failed
        # passing timeout < 0 means check worker status and return immediately
        state = worker_group.state
        worker_pids = {w.id for w in worker_group.workers}
        pc_pids = set(self._process_context.pids())
        if worker_pids != pc_pids:
            log.error(f"[{role}] worker pids do not match process_context pids")
            return WorkerState.UNKNOWN

        try:
            if self._process_context.join(timeout=-1):
                state = WorkerState.SUCCEEDED
            else:
                state = WorkerState.HEALTHY
        except Exception as e:
            log.error(f"[{role}] Worker set failed", exc_info=e)
            state = WorkerState.FAILED

        return state
