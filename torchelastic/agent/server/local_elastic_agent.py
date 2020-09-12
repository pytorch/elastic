#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

import torch.multiprocessing as mp
from torchelastic.agent.server.api import (
    MonitorResult,
    SimpleElasticAgent,
    Worker,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torchelastic.metrics.api import prof
from torchelastic.utils.logging import get_logger


log = get_logger()


class _DistInfo:
    """
    Container for information required to create a torch process group.
    To be created on the agent's process and passed to the worker sub-process.
    Hence this object needs to be a pure data object with no state and
    preferably only primitive member variables
    """

    __slots__ = [
        "rank",
        "group_rank",
        "role_rank",
        "local_world_size",
        "role_world_size",
        "world_size",
        "master_addr",
        "master_port",
        "restart_count",
        "max_restarts",
        "run_id",
        "role_name",
    ]

    def __init__(
        self,
        rank: int,
        group_rank: int,
        role_rank: int,
        local_world_size: int,
        role_world_size: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        restart_count: int,
        max_restarts: int,
        run_id: str,
        role_name: str,
    ):
        self.rank = rank
        self.group_rank = group_rank
        self.local_world_size = local_world_size
        self.role_rank = role_rank
        self.world_size = world_size
        self.role_world_size = role_world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.restart_count = restart_count
        self.max_restarts = max_restarts
        self.run_id = run_id
        self.role_name = role_name


def _wrap(local_rank, ret_val_queue, dist_infos, fn, args):
    import faulthandler

    try:
        faulthandler.enable(all_threads=True)
    except Exception as e:
        log.warn(
            "Unable to enable fault handler. Failure signals on worker process will not dump tracebacks",
            exc_info=e,
        )

    info = dist_infos[local_rank]
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(info.rank)
    os.environ["GROUP_RANK"] = str(info.group_rank)
    os.environ["ROLE_RANK"] = str(info.role_rank)
    os.environ["ROLE_NAME"] = info.role_name
    os.environ["LOCAL_WORLD_SIZE"] = str(info.local_world_size)
    os.environ["WORLD_SIZE"] = str(info.world_size)
    os.environ["ROLE_WORLD_SIZE"] = str(info.role_world_size)
    os.environ["MASTER_ADDR"] = info.master_addr
    os.environ["MASTER_PORT"] = str(info.master_port)
    os.environ["TORCHELASTIC_RESTART_COUNT"] = str(info.restart_count)
    os.environ["TORCHELASTIC_MAX_RESTARTS"] = str(info.max_restarts)
    os.environ["TORCHELASTIC_RUN_ID"] = info.run_id
    ret = fn(*args)
    ret_val_queue.put((info.rank, ret))


class LocalElasticAgent(SimpleElasticAgent):
    """
    An implementation of :py:class:`torchelastic.agent.server.ElasticAgent`
    that handles host-local workers.
    This agent is deployed per host and is configured to spawn ``n`` workers.
    When using GPUs, ``n`` maps to the number of GPUs available on the host.

    The local agent does not communicate to other local agents deployed on
    other hosts, even if the workers may communicate inter-host. The worker id
    is interpreted to be a local process. The agent starts and stops all worker
    processes as a single unit.

    The worker function and argument passed to the worker function must be
    python multiprocessing compatible. To pass multiprocessing data structures
    to the workers you may create the data structure in the same multiprocessing
    context as the specified ``start_method`` and pass it as a function argument.

    The exit_barrier_timeout specifies the amount of time (in seconds) to wait
    for other agents to finish. This acts as a safety net to handle cases where
    workers finish at different times, to prevent agents from viewing workers
    that finished early as a scale-down event. It is strongly advised that the
    user code deal with ensuring that workers are terminated in a synchronous
    manner rather than relying on the exit_barrier_timeout.

    Example

    ::

        def trainer(shared_queue):
            pass

        def main():
            start_method="spawn"
            shared_queue= multiprocessing.get_context(start_method).Queue()
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        fn=trainer,
                        args=(shared_queue,),
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            agent.run()
    """

    def __init__(
        self, spec: WorkerSpec, start_method="spawn", exit_barrier_timeout: float = 300
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        # pyre-ignore[8]: Attribute has type `ProcessContext`; used as `None`.
        self._process_context: mp.ProcessContext = None
        # a queue that holds return values for each worker fn
        # each element of the queue is a tuple (rank, ret_val)
        self._ret_val_queue = None

    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        for proc in self._process_context.processes:
            if proc.is_alive():
                proc.terminate()
            proc.join()

    @prof
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
                worker_group.group_rank,
                worker.role_rank,
                spec.local_world_size,
                worker.role_world_size,
                worker.world_size,
                master_addr,
                master_port,
                restart_count,
                spec.max_restarts,
                spec.rdzv_handler.get_run_id(),
                spec.role,
            )

        self._ret_val_queue = mp.get_context(self._start_method).Queue()
        self._process_context = mp.start_processes(
            fn=_wrap,
            args=(self._ret_val_queue, dist_infos, spec.fn, spec.args),
            nprocs=spec.local_world_size,
            join=False,
            daemon=False,
            start_method=self._start_method,
        )

        return {
            local_rank: pid
            for local_rank, pid in enumerate(self._process_context.pids())
        }

    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> MonitorResult:
        role = worker_group.spec.role

        # torch process context join() isn't really a join in the
        # traditional sense, it returns True if all the workers have
        # successfully finished, False if some/all are still running
        # and throws an Exception if some/all of them failed
        # passing timeout < 0 means check worker status and return immediately

        worker_pids = {w.id for w in worker_group.workers}
        pc_pids = set(self._process_context.pids())
        if worker_pids != pc_pids:
            log.error(f"[{role}] worker pids do not match process_context pids")
            return MonitorResult(WorkerState.UNKNOWN)

        try:
            if self._process_context.join(timeout=-1):
                # copy ret_val_queue into a map
                ret_vals = {}
                assert worker_group.spec.local_world_size == self._ret_val_queue.qsize()
                for _ in range(self._ret_val_queue.qsize()):
                    (rank, out) = self._ret_val_queue.get()
                    ret_vals[rank] = out
                self._ret_val_queue = None
                return MonitorResult(WorkerState.SUCCEEDED, ret_vals)
            else:
                return MonitorResult(WorkerState.HEALTHY)
        except Exception as e:
            log.exception(f"[{role}] Worker group failed")
            return MonitorResult(
                WorkerState.FAILED,
                exceptions={w.global_rank: e for w in worker_group.workers},
            )
