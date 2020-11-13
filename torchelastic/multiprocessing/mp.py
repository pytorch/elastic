#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.multiprocessing as mp
from torchelastic.multiprocessing.api import BaseProcessContext, ProcessGroupResult
from torchelastic.multiprocessing.errors import (
    exec_fn,
    get_error_file,
    get_failed_result,
)


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MpParameters:
    """
    Specifies parameters that are used for launching processes via torch.mp
    """

    fn: Callable
    args: Tuple


def _wrap(
    local_rank: int,
    error_files,
    params: List[MpParameters],
    ret_val_queues: Dict[int, mp.SimpleQueue],
):
    os.environ["TORCHELASTIC_ERROR_FILE"] = error_files[local_rank]
    proc_params = params[local_rank]
    ret = exec_fn(proc_params.fn, args=(local_rank, *proc_params.args))
    ret_val_queue = ret_val_queues[local_rank]
    # Note: ret_val_queue will always contain a single element.
    ret_val_queue.put(ret)


class MpProcessContext(BaseProcessContext):
    """
    A wrapper around torch.multiprocessing that implements a common interface of
    launching and controlling processes.
    """

    def __init__(
        self, mp_process_context: mp.ProcessContext, ret_val_queues, run_id: int = 0
    ):
        self._process_context: mp.ProcessContext = mp_process_context
        self._ret_val_queues = ret_val_queues
        self._worker_ret_vals = {}
        self.run_id = run_id

    def wait(self, timeout: Optional[float] = None) -> Optional[ProcessGroupResult]:
        deadline, period = self._get_deadline_and_period(timeout)
        try:
            while deadline > time.time():
                self._worker_ret_vals.update(self._try_collect_outputs())
                if self._process_context.join(period):
                    break
            if not self._process_context.join(-1):
                return None
        except mp.ProcessExitedException as e:
            # Print exception message to the log otherwise logs would not have it
            logger.exception(f"Process {e.pid} failed")
            return ProcessGroupResult(
                return_values={},
                failure=get_failed_result(
                    e.error_index, e.pid, e.exit_code, self.run_id
                ),
            )
        except mp.ProcessRaisedException as e:
            # Print exception message to the log otherwise logs would not have it
            logger.exception(f"Process {e.pid} failed")
            return ProcessGroupResult(
                return_values={},
                failure=get_failed_result(e.error_index, e.pid, 1, self.run_id),
            )

        self._worker_ret_vals.update(self._try_collect_outputs())
        if len(self._worker_ret_vals) != len(self._process_context.processes):
            ranks = self._collect_workers_ranks_without_return_values()
            raise RuntimeError(
                f"Workers: {ranks} did not return any values, this should never happend and indicates bug"
            )
        return ProcessGroupResult(
            return_values=self._worker_ret_vals,
        )

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
    params: List[MpParameters], start_method: str = "spawn", run_id: int = 0
) -> MpProcessContext:
    r"""
    Launches processes and returns context object. Users can use that object
    to wait for processes results.
    """
    nprocs = len(params)
    ret_val_queues: Dict[int, mp.SimpleQueue] = {
        i: mp.get_context(start_method).SimpleQueue() for i in range(0, nprocs)
    }
    error_files = [
        get_error_file(local_rank, run_id) for local_rank in range(len(params))
    ]
    mp_proc_context = mp.start_processes(
        nprocs=nprocs,
        fn=_wrap,
        args=(error_files, params, ret_val_queues),
        join=False,
        daemon=False,
        start_method=start_method,
    )
    return MpProcessContext(mp_proc_context, ret_val_queues, run_id)
