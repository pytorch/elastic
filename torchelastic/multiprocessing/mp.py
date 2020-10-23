#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.multiprocessing as mp
from torchelastic.multiprocessing.api import BaseProcessContext, expire


@dataclass
class MpParameters:
    """
    Specifies parameters that are used for launching processes via torch.mp
    """

    fn: Callable
    args: Tuple


def _wrap(
    local_rank: int,
    params: List[MpParameters],
    ret_val_queues: Dict[int, mp.SimpleQueue],
):
    proc_params = params[local_rank]
    ret = proc_params.fn(local_rank, *proc_params.args)
    ret_val_queue = ret_val_queues[local_rank]
    # Note: ret_val_queue will always contain a single element.
    ret_val_queue.put(ret)


class MpProcessContext(BaseProcessContext):
    """
    A wrapper around torch.multiprocessing that implements a common interface of
    launching and controlling processes.
    """

    def __init__(self, mp_process_context: mp.ProcessContext, ret_val_queues):
        self._process_context: mp.ProcessContext = mp_process_context
        self._ret_val_queues = ret_val_queues
        self._worker_ret_vals = {}

    def wait(self, timeout: Optional[float] = None) -> Optional[Dict[int, Any]]:
        def _wait(deadline, period) -> bool:
            self._worker_ret_vals.update(self._try_collect_outputs())
            return self._process_context.join(period)

        expire(fn=_wait, timeout=timeout)

        if not self._process_context.join(-1):
            return None
        self._worker_ret_vals.update(self._try_collect_outputs())
        if len(self._worker_ret_vals) != len(self._process_context.processes):
            ranks = self._collect_workers_ranks_without_return_values()
            raise RuntimeError(
                f"Workers: {ranks} did not return any values, this should never happend and indicates bug"
            )
        return self._worker_ret_vals

    def pids(self) -> List[int]:
        return self._process_context.pids()

    def terminate(self) -> None:
        for proc in self._process_context.processes:
            if proc.is_alive():
                proc.terminate()
            proc.join()

    def _collect_workers_ranks_without_return_values(self):
        # returns ranks of workers which queues did not return any values.
        # This normally should never happen and indicates that there is a
        # bug in the code.
        ranks = []
        for idx in range(0, len(self._process_context.processes)):
            if idx not in self._worker_ret_vals:
                ranks.append(idx)
        return ranks

    def _try_collect_outputs(self) -> Dict[int, Any]:
        out_vals = {}
        for idx in range(len(self._process_context.processes)):
            ret_queue = self._ret_val_queues[idx]
            if not ret_queue.empty():
                out = ret_queue.get()
                out_vals[idx] = out
        return out_vals


def start_processes(
    params: List[MpParameters], start_method: str = "spawn"
) -> MpProcessContext:
    r"""
    Launches processes and returns context object. Users can use that object
    to wait for processes results.
    """
    nprocs = len(params)
    ret_val_queues: Dict[int, mp.SimpleQueue] = {
        i: mp.get_context(start_method).SimpleQueue() for i in range(0, nprocs)
    }
    mp_proc_context = mp.start_processes(
        nprocs=nprocs,
        fn=_wrap,
        args=(params, ret_val_queues),
        join=False,
        daemon=False,
        start_method=start_method,
    )
    return MpProcessContext(mp_proc_context, ret_val_queues)
