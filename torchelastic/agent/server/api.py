#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import functools
import json
import socket
import time
from contextlib import closing
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import torchelastic.rendezvous as rdzv
import torchelastic.utils.store as store_util
from torchelastic.metrics import prof, put_metric
from torchelastic.utils.logging import get_logger


_TERMINAL_STATE_SYNC_ID = "torchelastic/agent/terminal_state"

DEFAULT_ROLE = "default"
log = get_logger()


class WorkerSpec:
    """
    Contains blueprint information about a particular type of worker.
    For a given role, there must only exist a single worker spec.
    Worker spec is expected to be homogenous across all nodes (machine),
    that is each node runs the same number of workers for a particular spec.
    """

    __slots__ = [
        "role",
        "local_world_size",
        "fn",
        "args",
        "rdzv_handler",
        "max_restarts",
        "monitor_interval",
        "master_port",
    ]

    def __init__(
        self,
        role: str,
        local_world_size: int,
        fn: Callable,
        args: Tuple,
        rdzv_handler: rdzv.RendezvousHandler,
        max_restarts: int = 3,
        monitor_interval: float = 30.0,
        master_port=None,
    ):
        r"""

        Arguments:
            role (str): user-defined role for the workers with this spec
            local_world_size (int): number local workers to run
            fn (Callable): worker main entry point function
            args (Tuple): arguments to pass to ``fn(args)``
            rdzv_handler (RendezvousHandler): handles rdzv for this set of workers
            max_restarts (int): number of max retries for the workers
            monitor_interval (int): monitor status of workers every ``n`` seconds
            master_port (int): fixed port to run the c10d store on rank 0
                               if not specified then will chose a random free port
        """

        assert local_world_size > 0
        assert max_restarts >= 0
        assert monitor_interval > 0

        # Note: role is not used for data parallel, every worker has the same role
        # wiring it in to handle more elaborate situations later
        self.role = role
        self.local_world_size = local_world_size
        self.fn = fn
        self.args = args
        self.rdzv_handler = rdzv_handler
        self.max_restarts = max_restarts
        self.monitor_interval = monitor_interval
        self.master_port = master_port


class Worker:
    """
    Represents a worker instance. Contrast this with ``WorkerSpec`` that
    represents the specifications of a worker. A ``Worker`` is created from
    a ``WorkerSpec``. A ``Worker`` is to a ``WorkerSpec`` as an object is to
    a class.

    The ``id`` of the worker is interpreted
    by the specific implementation of ``ElasticAgent``. For a local
    agent, it could be the ``pid (int)`` of the worker, for a remote
    agent it could be encoded as ``host:port (string)``.

    Arguments:
        id (Any): uniquely identifies a worker (interpreted by the agent)
        local_rank (int): local rank of the worker
        global_rank (int): global rank of the worker
        role_rank (int): rank of the worker across all workers that have the same role
        world_size (int): number of workers (globally)
        role_world_size (int): number of workers that have the same role
    """

    __slots__ = [
        "id",
        "local_rank",
        "global_rank",
        "role_rank",
        "world_size",
        "role_world_size",
    ]

    def __init__(
        self,
        local_rank: int,
        global_rank: int = -1,
        role_rank: int = -1,
        world_size: int = -1,
        role_world_size: int = -1,
    ):
        # unique identifier for this worker
        self.id: Any = None

        # rank of the worker among workers with the same role being monitored
        # by the same ``agent`` instance.
        self.local_rank: int = local_rank

        #  rank of the worker among all the workers across all roles
        #  across all ``agent`` instances.
        #  Global rank is not stable between re-rendezvous.
        self.global_rank: int = global_rank

        #  rank of the worker among all the workers with the same role
        #  across all ``agent`` instances.
        #  Global rank is not stable between re-rendezvous.
        self.role_rank: int = role_rank

        # total number of workers (globally). Due to elasticity
        # the world size may change between re-rendezvous.
        self.world_size: int = world_size

        # total number of workers that share the same role. Due to elasticity
        # the role world size may change between re-rendezvous.
        self.role_world_size: int = role_world_size

    def __str__(self):
        return (
            f"local_rank={self.local_rank},global_rank={self.global_rank}"
            f",role_rank={self.role_rank},world_size={self.world_size}"
            f",role_world_size={self.role_world_size}"
        )

    def __repr__(self):
        return str(self)


class WorkerState(Enum):
    """
    State of the ``WorkerGroup``. Workers in a worker group change state as a unit.
    If a single worker in a worker group fails the entire set is considered
    failed::

      UNKNOWN - agent lost track of worker group state, unrecoverable
      INIT - worker group object created not yet started
      HEALTHY - workers running and healthy
      UNHEALTHY - workers running and unhealthy
      STOPPED - workers stopped (interruped) by the agent
      SUCCEEDED - workers finished running (exit 0)
      FAILED - workers failed to successfully finish (exit !0)


    A worker group starts from an initial ``INIT`` state,
    then progresses to ``HEALTHY`` or ``UNHEALTHY`` states,
    and finally reaches a terminal ``SUCCEEDED`` or ``FAILED`` state.

    Worker groups can be interrupted and temporarily put into ``STOPPED`` state
    by the agent. Workers in ``STOPPED`` state are scheduled to be restarted
    in the near future by the agent. Some examples of workers being put into
    ``STOPPED`` state are:

    1. Worker group failure|unhealthy observed
    2. Membership change detected

    When actions (start, stop, rdzv, retry, etc) on worker group fails
    and results in the action being partially applied to the worker group
    the state will be ``UNKNOWN``. Typically this happens on uncaught/unhandled
    exceptions during state change events on the agent. The agent is not
    expected to recover worker groups in ``UNKNOWN`` state and is better off
    self terminating and allowing the job manager to retry the node.
    """

    UNKNOWN = 0
    INIT = 1
    HEALTHY = 2
    UNHEALTHY = 4
    STOPPED = 8
    SUCCEEDED = 16
    FAILED = 32

    @staticmethod
    def is_running(state: "WorkerState") -> bool:
        """
        Returns:
             `` True`` if the worker state represents workers still running
              (e.g. that the process exists but not necessarily healthy).
        """
        return state in {WorkerState.HEALTHY, WorkerState.UNHEALTHY}


class WorkerGroup:
    """
    Represents the set of ``Worker`` instances for the given ``WorkerSpec``
    managed by ``ElasticAgent``. Whether the worker group contains cross
    instance workers or not depends on the implementation of the agent.
    """

    __slots__ = ["spec", "workers", "store", "group_rank", "group_world_size", "state"]

    def __init__(self, spec: WorkerSpec):
        self.spec = spec
        self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]

        # assigned after rdzv
        self.store = None
        self.group_rank = None
        self.group_world_size = None

        self.state = WorkerState.INIT


class _RoleInstanceInfo:
    """
    The class is used by the agent to exchange the information with other agents.
    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """

    __slots__ = ["role", "rank", "local_world_size"]

    def __init__(self, role: str, rank: int, local_world_size: int):
        r"""

        Arguments:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """

        self.role = role
        self.rank = rank
        self.local_world_size = local_world_size

    def serialize(self) -> bytes:
        dict_data = {
            "role": self.role,
            "rank": self.rank,
            "local_world_size": self.local_world_size,
        }
        return json.dumps(dict_data).encode(encoding="UTF-8")

    @staticmethod
    def deserialize(data: bytes):
        dict_data = json.loads(data.decode(encoding="UTF-8"))
        return _RoleInstanceInfo(
            dict_data["role"], dict_data["rank"], dict_data["local_world_size"]
        )

    @staticmethod
    def compare(obj1, obj2) -> int:
        if obj1.role == obj2.role:
            return obj1.rank - obj2.rank
        elif obj1.role > obj2.role:
            return 1
        else:
            return -1

    @staticmethod
    def find_role_boundaries(roles_infos: List, role: str) -> Tuple[int, int]:
        start_idx, end_idx = -1, -1
        for idx, role_info in enumerate(roles_infos):
            if role_info.role == role:
                if start_idx == -1:
                    start_idx = idx
                end_idx = idx
        return (start_idx, end_idx)


class MonitorResult:
    """
    Returned by the agent's ``_monitor_workers`` API. A holder object
    that holds information about the monitoring results.
    The ``ret_vals`` and ``exceptions`` field map each worker's
    return value (output) and exceptions (if any) accordingly by
    the workers global rank.

    ``state = SUCCEEDED`` will have ``ret_val``.
    ``state = FAILED`` will have ``exceptions``.
    For other states both these fields will be empty.
    """

    __slots__ = ["state", "ret_vals", "exceptions"]

    def __init__(
        self,
        state: WorkerState,
        # pyre-fixme[9]: ret_vals has type `Dict[int, typing.Any]`; used as `None`.
        ret_vals: Dict[int, Any] = None,
        # pyre-fixme[9]: exceptions has type `Dict[int, Exception]`; used as `None`.
        exceptions: Dict[int, Exception] = None,
    ):
        self.state = state
        self.ret_vals = ret_vals
        self.exceptions = exceptions


class WorkerGroupFailureException(Exception):
    """
    Thrown when the agent cannot or has given up trying to run the workers.
    This is typically thrown:

    1. Exceeded ``max_restarts``.
    2. Workers fail with errors that are deemed ``NonRestartable``

    When constructing this exception the underlying worker exceptions
    are provided as a map of the worker's global rank to the exception.
    """

    def __init__(self, msg: str, worker_excs: Dict[int, Exception]):
        super().__init__(msg)
        self._worker_excs = worker_excs

    def get_worker_exceptions(self) -> Dict[int, Exception]:
        return self._worker_excs


def _get_socket_with_port() -> socket.socket:
    """
    Returns a free port on localhost that is "reserved" by binding a temporary
    socket on it. Close the socket before passing the port to the entity
    that requires it. Usage example

    ::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    """

    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            log.info("Socket creation attempt failed.", exc_info=e)
    raise RuntimeError("Failed to create a socket")


def _get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


class ElasticAgent(abc.ABC):
    """
    Agent process responsible for managing one or more worker processes.
    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user's job placement
    preferences. For instance, to run a distributed training job on GPU with
    8 trainers (one per GPU) one can:

    1. Use 8 x single GPU instances, place an agent per instance, managing
       1 worker per agent.
    2. Use 4 x double GPU instances, place an agent per instance, managing
       2 workers per agent.
    3. Use 2 x quad GPU instances, place an agent per instance, managing
       4 workers per agent.
    4. Use 1 x 8 GPU instance, place an agent per instance, managing
       8 workers per agent.

    Usage
    ::

     try:
         results = agent.run()
         return results[0] # return rank 0's results
     except WorkerGroupFailureException as e:
         exceptions = e.get_worker_exceptions()
         log.exception(f"worker 0 failed with: {exceptions[0]}")
     except Exception as e:
         log.exception(f"error while running agent")

    """

    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> Dict[int, Any]:
        """
        Runs the agent, retrying the worker group on failures up to
        ``max_restarts``.

        Returns:
            The return values for each worker mapped by the worker's global rank.
            Empty if workers have void signature.

        Raises:
            WorkerGroupFailureException - workers did not successfully run
            Exception - any other failures NOT related to worker process
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """
        Returns:
            The ``WorkerGroup`` for the given ``role``.
            Note that the worker group is a mutable object and hence in a
            multi-threaded/process environment it may change state.
            Implementors are encouraged (but not required) to return
            a defensive read-only copy.
        """
        raise NotImplementedError()


class SimpleElasticAgent(ElasticAgent):
    """
    An ``ElasticAgent`` that manages workers (``WorkerGroup``)
    for a single ``WorkerSpec`` (e.g. one particular type of worker role).
    """

    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300):
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = self._worker_group.spec.max_restarts
        self._store = None
        self._exit_barrier_timeout = exit_barrier_timeout

    # pyre-fixme[14]: `get_worker_group` overrides method defined in `ElasticAgent`
    #  inconsistently.
    def get_worker_group(self) -> WorkerGroup:
        # TODO return an RO copy (need to create an ROWorkerGroup and ROWorkerSpec
        # since both these classes contain non-pure-data pointers - e.g. rdzv_handler)
        return self._worker_group

    @abc.abstractmethod
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        r"""
        Starts ``worker_group.spec.local_world_size`` number of workers
        according to worker spec for the worker group .

        Returns a map of ``local_rank`` to worker ``id``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _stop_workers(self, worker_group: WorkerGroup) -> None:
        r"""
        Stops all workers in the given worker group. Implementors
        must deal with workers in all states defined by ``WorkerState``.
        That is, it must gracefully handle stopping non-existent workers,
        unhealthy (stuck) workers, etc.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _monitor_workers(self, worker_group: WorkerGroup) -> MonitorResult:
        r"""
        Checks on the workers for the ``worker_group`` and returns
        the new state of the worker group.
        """
        raise NotImplementedError()

    @staticmethod
    def _set_master_addr_port(store, master_port):
        if master_port is None:
            sock = _get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]

        store.set("MASTER_ADDR", _get_fq_hostname().encode(encoding="UTF-8"))
        store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))

    @staticmethod
    def _get_master_addr_port(store) -> Tuple[str, int]:
        master_addr = store.get("MASTER_ADDR").decode(encoding="UTF-8")
        master_port = int(store.get("MASTER_PORT").decode(encoding="UTF-8"))
        return (master_addr, master_port)

    @prof
    def _rendezvous(self, worker_group: WorkerGroup) -> None:
        r"""
        Runs rendezvous for the workers specified by worker spec.
        Assigns workers a new global rank and world size.
        Updates the rendezvous store for the worker group.
        """

        spec = worker_group.spec

        store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
        self._store = store

        workers = self._assign_worker_ranks(store, group_rank, group_world_size, spec)
        worker_group.workers = workers
        worker_group.store = store
        worker_group.group_rank = group_rank
        worker_group.group_world_size = group_world_size

        if group_rank == 0:
            self._set_master_addr_port(store, spec.master_port)
        master_addr, master_port = self._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts
        workers_info = {
            "local_ranks": [worker.local_rank for worker in workers],
            "global_ranks": [worker.global_rank for worker in workers],
            "role_ranks": [worker.role_rank for worker in workers],
            "world_size": workers[0].world_size,
            "role_world_size": workers[0].role_world_size,
        }
        log.info(
            f"[{spec.role}] Rendezvous complete for workers.\n"
            f"Result:\n"
            f"\trestart_count={restart_count}\n"
            f"\tgroup_rank={group_rank}\n"
            f"\tgroup_world_size={group_world_size}\n"
            f"\tmaster_addr={master_addr}\n"
            f"\tmaster_port={master_port}\n"
            f"\tworkers={workers_info}\n"
        )

    def _get_ranks(
        self,
        role_infos: List[_RoleInstanceInfo],
        role_idx: int,
        start_idx: int = 0,
        end_idx: int = -1,
    ) -> Tuple[int, List[int]]:
        if end_idx == -1:
            end_idx = len(role_infos)
        prefix_sum = 0
        total_sum = 0
        for idx in range(start_idx, end_idx):
            if role_idx > idx:
                prefix_sum += role_infos[idx].local_world_size
            total_sum += role_infos[idx].local_world_size
        return (
            total_sum,
            list(range(prefix_sum, prefix_sum + role_infos[role_idx].local_world_size)),
        )

    @prof
    def _assign_worker_ranks(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> List[Worker]:
        """
        Determines proper ranks for worker processes. The rank assignment
        is done according to the following algorithm:

        1. Each agent writes its configuration(group_rank, group_world_size
           , num_workers) to the common store.
        2. Each agent retrieves configuration for all agents
           and performs two level sort using role and rank.
        3. Determine the global rank: the global rank of the workers for the current
           agent is the offset of the infos array up to group_rank of the agent.
           The offset is computed as a sum of local_world_size of all agents that
           have rank less than the group_rank. The workers would have the ranks:
           [offset, offset+local_world_size)
        4. Determine the role rank: The role rank is determined using the algorithms
           in the point 3 with the exception that the offset is done from the first
           agent that has the same role as current one and has the minimum group rank.
        """

        role_infos = self._share_and_gather(store, group_rank, group_world_size, spec)
        my_role_info = role_infos[group_rank]
        worker_world_size, worker_global_ranks = self._get_ranks(role_infos, group_rank)
        role_infos = sorted(
            role_infos, key=functools.cmp_to_key(_RoleInstanceInfo.compare)
        )
        role_start_idx, role_end_idx = _RoleInstanceInfo.find_role_boundaries(
            role_infos, my_role_info.role
        )
        role_pos = next(
            idx
            for idx, role_info in enumerate(role_infos)
            if _RoleInstanceInfo.compare(role_info, my_role_info) == 0
        )
        role_world_size, role_ranks = self._get_ranks(
            role_infos, role_pos, role_start_idx, role_end_idx + 1
        )
        workers = []
        for ind in range(spec.local_world_size):
            worker = Worker(
                local_rank=ind,
                global_rank=worker_global_ranks[ind],
                role_rank=role_ranks[ind],
                world_size=worker_world_size,
                role_world_size=role_world_size,
            )
            workers.append(worker)
        return workers

    def _share_and_gather(
        self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
    ) -> List:
        agent_role_info = _RoleInstanceInfo(
            spec.role, group_rank, spec.local_world_size
        )
        key_prefix = "torchelastic/role_info"
        agent_config_enc = agent_role_info.serialize()
        role_infos_bytes = store_util.synchronize(
            store, agent_config_enc, group_rank, group_world_size, key_prefix
        )
        role_infos = [
            _RoleInstanceInfo.deserialize(role_info_bytes)
            for role_info_bytes in role_infos_bytes
        ]
        return role_infos

    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        r"""
        Starts a fresh set of workers for the worker_group.
        Essentially a rendezvous followed by a start_workers.

        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        role = worker_group.spec.role
        log.info(f"[{role}] Rendezvous'ing worker group")

        # TODO after stopping workers, wait at least monitor_interval*2 for
        # workers on different nodes to fail on a collective op before waiting
        # on the rdzv barrier, this way we ensure that nodes enter rdzv
        # at around the same time and reduce false positive rdzv timeout errors
        self._rendezvous(worker_group)

        log.info(f"[{role}] Starting worker group")
        worker_ids = self._start_workers(worker_group)
        for local_rank, id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = id

        worker_group.state = WorkerState.HEALTHY

    @prof
    def _restart_workers(self, worker_group: WorkerGroup) -> None:
        """
        Restarts (stops, rendezvous, starts) all local workers in the group.
        """

        role = worker_group.spec.role
        log.info(f"[{role}] Stopping worker group")
        self._stop_workers(worker_group)
        worker_group.state = WorkerState.STOPPED
        self._initialize_workers(worker_group)

    @prof
    def run(self, role: str = DEFAULT_ROLE) -> Dict[int, Any]:
        is_failed = False
        try:
            return self._invoke_run(role)
        except Exception as e:
            is_failed = True
            raise
        finally:
            self._record_metrics(is_failed)

    def _record_metrics(self, is_failed: bool = False):
        self._record_flakiness_metric(is_failed)
        spec = self._worker_group.spec
        restarts_happened = self._remaining_restarts != spec.max_restarts
        put_metric(f"workers.{spec.role}.run_total", 1)
        self._record_metric_with_condition(
            "run_success_with_retries", not is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_success_no_retries", not is_failed and not restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_with_retries", is_failed and restarts_happened
        )
        self._record_metric_with_condition(
            "run_failed_no_retries", is_failed and not restarts_happened
        )

    def _record_metric_with_condition(self, metric_name, condition):
        spec = self._worker_group.spec
        if condition:
            put_metric(f"workers.{spec.role}.{metric_name}", 1)
        else:
            put_metric(f"workers.{spec.role}.{metric_name}", 0)

    def _record_flakiness_metric(self, is_failed: bool = False):
        if is_failed:
            if not isinstance(is_failed, WorkerGroupFailureException):
                # Only user code can contribute into flakiness score
                return
            flakiness = 100.0
        else:
            spec = self._worker_group.spec
            flakiness = 100.0 - 100.0 * (self._remaining_restarts + 1) / (
                spec.max_restarts + 1
            )
        spec = self._worker_group.spec

        put_metric(f"workers.{spec.role}.flakiness", int(flakiness))

    def _invoke_run(self, role: str = DEFAULT_ROLE) -> Dict[int, Any]:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        log.info(f"[{role}] starting workers for function: {spec.fn.__name__}")

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            monitor_result = self._monitor_workers(self._worker_group)
            state = monitor_result.state
            self._worker_group.state = state

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                log.info(
                    f"[{role}] worker group successfully finished."
                    f" Waiting {self._exit_barrier_timeout} seconds for other agents to finish."
                )
                self._exit_barrier()
                return monitor_result.ret_vals
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    log.info(
                        f"[{role}] Worker group {state.name}. "
                        f"{self._remaining_restarts}/{spec.max_restarts} attempts left;"
                        f" will restart worker group"
                    )
                    self._remaining_restarts -= 1
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    self._exit_barrier()
                    raise WorkerGroupFailureException(
                        f"[{role}] exceeded max_restarts={spec.max_restarts}",
                        monitor_result.exceptions,
                    )
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    log.info(
                        f"[{role}] Detected {num_nodes_waiting} "
                        f"new nodes from group_rank={group_rank}; "
                        f"will restart worker group"
                    )
                    self._restart_workers(self._worker_group)
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")

    def _exit_barrier(self):
        """
        Wait for ``exit_barrier_timeout`` seconds for all agents to finish
        executing their local workers (either successfully or not). This
        acts as a safety guard against user scripts that terminate at different
        times. This barrier keeps the agent process alive until all workers finish.
        """
        log.info(
            f"Local worker group finished ({self._worker_group.state}). "
            f"Waiting {self._exit_barrier_timeout} seconds for other agents to finish"
        )
        start = time.time()
        try:
            store_util.barrier(
                self._store,
                self._worker_group.group_rank,
                self._worker_group.group_world_size,
                key_prefix=_TERMINAL_STATE_SYNC_ID,
                barrier_timeout=self._exit_barrier_timeout,
            )
            log.info(
                f"Done waiting for other agents. Elapsed: {time.time() - start} seconds"
            )
        except Exception:
            log.exception(
                f"Error waiting on exit barrier. Elapsed: {time.time() - start} seconds"
            )
