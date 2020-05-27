#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable

import torch.distributed.rpc as torch_rpc


class WorkerInfo:
    __slots__ = [
        "name",  # name (same as torch.distributed.rpc.WorkerInfo.name)
        "id",  # torch rpc id (same as torch.distributed.rpc.WorkerInfo.id)
        "rank",  # global rank
        "role_rank",  # role rank
    ]

    def __init__(self, name, id, rank, role_rank):
        self.name = name
        self.id = id
        self.rank = rank
        self.role_rank = role_rank


class RoleInfo:
    __slots__ = [
        "name",
        "role_world_size",
        "local_world_size",
        "worker_infos",  # List[WorkerInfo]
    ]

    def __init__(self, name, role_world_size, local_world_size, worker_infos):
        self.name = name
        self.role_world_size = role_world_size
        self.local_world_size = local_world_size
        self.worker_infos = worker_infos


###########################
# Initializers
###########################


def init_app(
    role: str,
    # pyre-fixme[11]: Annotation `BackendType` is not defined as a type.
    backend: torch_rpc.backend_registry.BackendType,
    # pyre-fixme[11]: Annotation `RpcBackendOptions` is not defined as a type.
    backend_options: torch_rpc.RpcBackendOptions,
):
    # TODO placeholder; implement
    pass


###########################
# Info Accessors
###########################


def get_worker_names(role: str) -> Iterable[str]:
    """
    Returns all the worker names for the specified role.
    """
    return [info.name for info in get_role_info(role).worker_infos]


def get_role_info(role: str) -> RoleInfo:
    """
    Returns the role information.
    """
    # TODO placeholder; implement
    return RoleInfo(name=role, role_world_size=0, local_world_size=0, worker_infos=[])


def get_all_roles() -> Iterable[str]:
    """
    Returns the names of all roles in this application.
    """
    # TODO placeholder; implement
    return []


###########################
# Futures
###########################


def wait_all(futures):
    # TODO placeholder implementation; make better
    if isinstance(futures, Dict):
        results = {}
        for name, fut in futures.items():
            results[name] = fut.wait()
    else:
        results = []
        for fut in futures.items():
            results.append(fut.wait())
    return results


###########################
# RPC APIs
###########################


def rpc_sync_on_role(
    role: str, func, args=None, kwargs=None, timeout=None
) -> Dict[str, Any]:
    # TODO placeholder; implement
    futs = rpc_async_on_role(role, func, args, kwargs, timeout)
    return wait_all(futs)


def rpc_async_on_role(
    role: str, func, args=None, kwargs=None, timeout=None
) -> Dict[str, torch_rpc.Future]:
    # can't use rpc.UNSET_RPC_TIMEOUT (only available in torch 1.6.0+)
    # reproduce the same behavior by getting the timeout if one is not passed
    if timeout is None:
        # pyre-fixme[16]: Module `torch_rpc` has no attribute `get_rpc_timeout`.
        timeout = torch_rpc.get_rpc_timeout()

    futures = {}
    for name in get_worker_names(role):
        fut = torch_rpc.rpc_async(name, func, args, kwargs, timeout)
        futures[name] = fut
    return futures


def remote_on_role(role: str, func, args=None, kwargs=None):
    rrefs = {}
    for name in get_worker_names(role):
        rref = torch_rpc.remote(name, func, args, kwargs)
        rrefs[name] = rref
    return rrefs


###########################
# Process Group APIs
###########################


def init_process_group():
    """
    Creates a process group amongst workers having the same role
    as the caller of this function.

    Usage::

      init_app(role="trainer", backend=BackendType.PROCESS_GROUP)
      init_process_group() # this worker joins the trainer process group
    """
    # TODO placeholder; implement
    pass
