#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from torchelastic.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    Worker,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torchelastic.metrics.api import prof
from torchelastic.multiprocessing import (
    BaseProcessContext,
    MpParameters,
    SubprocessParameters,
    start_processes,
    start_subprocesses,
)
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


def _get_worker_env(dist_info: _DistInfo, local_rank: int) -> Dict[str, str]:
    worker_env = {}
    worker_env["LOCAL_RANK"] = str(local_rank)
    worker_env["RANK"] = str(dist_info.rank)
    worker_env["GROUP_RANK"] = str(dist_info.group_rank)
    worker_env["ROLE_RANK"] = str(dist_info.role_rank)
    worker_env["ROLE_NAME"] = dist_info.role_name
    worker_env["LOCAL_WORLD_SIZE"] = str(dist_info.local_world_size)
    worker_env["WORLD_SIZE"] = str(dist_info.world_size)
    worker_env["ROLE_WORLD_SIZE"] = str(dist_info.role_world_size)
    worker_env["MASTER_ADDR"] = dist_info.master_addr
    worker_env["MASTER_PORT"] = str(dist_info.master_port)
    worker_env["TORCHELASTIC_RESTART_COUNT"] = str(dist_info.restart_count)
    worker_env["TORCHELASTIC_MAX_RESTARTS"] = str(dist_info.max_restarts)
    worker_env["TORCHELASTIC_RUN_ID"] = dist_info.run_id
    if "OMP_NUM_THREADS" in os.environ:
        worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    return worker_env


def _wrap(local_rank, dist_infos, fn, args):
    import faulthandler

    try:
        faulthandler.enable(all_threads=True)
    except Exception as e:
        log.warn(
            "Unable to enable fault handler. Failure signals on worker process will not dump tracebacks",
            exc_info=e,
        )

    worker_env = _get_worker_env(dist_infos[local_rank], local_rank)
    os.environ.update(worker_env)
    return fn(*args)


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

    The agent supports launching functions via torch.multiprocessing and
    launching arbitrary user commands via python subprocess.

    Example launching function

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

    Example launching command

    ::

        def main():
            start_method="spawn"
            spec = WorkerSpec(
                        role="trainer",
                        local_world_size=nproc_per_process,
                        cmd=["ls", "-la"]
                        ...<OTHER_PARAMS...>)
            agent = LocalElasticAgent(spec, start_method)
            agent.run()

    """

    def __init__(
        self, spec: WorkerSpec, start_method="spawn", exit_barrier_timeout: float = 300
    ):
        super().__init__(spec, exit_barrier_timeout)
        self._start_method = start_method
        self._process_context = None

    @prof
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        self._process_context.terminate()

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

        if spec.fn:
            self._process_context = self._start_mp(dist_infos, spec)
        else:
            self._process_context = self._start_sp(dist_infos, spec)

        return {
            local_rank: pid
            for local_rank, pid in enumerate(self._process_context.pids())
        }

    def _start_mp(
        self, dist_infos: Dict[int, _DistInfo], spec: WorkerSpec
    ) -> BaseProcessContext:
        proc_params = [
            MpParameters(fn=_wrap, args=(dist_infos, spec.fn, spec.args))
        ] * spec.local_world_size
        run_id = spec.max_restarts - self._remaining_restarts
        return start_processes(
            proc_params,
            start_method=self._start_method,
            run_id=run_id,
        )

    def _start_sp(
        self, dist_infos: Dict[int, _DistInfo], spec: WorkerSpec
    ) -> BaseProcessContext:
        proc_params = []
        for local_rank, dist_info in dist_infos.items():
            env = _get_worker_env(dist_info, local_rank)
            env.update(os.environ)
            proc_params.append(
                SubprocessParameters(
                    args=spec.cmd,
                    env=env,
                )
            )
        run_id = spec.max_restarts - self._remaining_restarts
        return start_subprocesses(proc_params, run_id=run_id)

    @prof
    def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
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
            return RunResult(state=WorkerState.UNKNOWN)

        proc_group_result = self._process_context.wait(timeout=1)
        if proc_group_result:
            if proc_group_result.is_failed():
                log.error(f"[{role}] Worker group failed")
                return RunResult(
                    state=WorkerState.FAILED,
                    return_values={},
                    failures={
                        w.global_rank: proc_group_result.failure
                        for w in worker_group.workers
                    },
                )
            else:
                # copy ret_val_queue into a map with a global ranks
                workers_ret_vals = {}
                for local_rank, ret_val in proc_group_result.return_values.items():
                    worker = worker_group.workers[local_rank]
                    workers_ret_vals[worker.global_rank] = ret_val
                return RunResult(
                    state=WorkerState.SUCCEEDED,
                    return_values=workers_ret_vals,
                    failures={},
                )
        else:
            return RunResult(state=WorkerState.HEALTHY)
