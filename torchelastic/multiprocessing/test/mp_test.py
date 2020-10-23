#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
from typing import Dict

import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException
from torchelastic.multiprocessing.mp import MpParameters, _wrap, start_processes
from torchelastic.test.test_utils import is_tsan


def run_compute(local_rank, mult=1) -> int:
    time.sleep(1)
    return local_rank * mult


def run_dummy(local_rank) -> None:
    time.sleep(1)


def run_with_wait(local_rank: int, wait=0) -> None:
    time.sleep(wait)


def fill_dict(local_rank, size: int) -> Dict[int, str]:
    out = {}
    for idx in range(0, size):
        out[idx] = f"test{idx}"
    return out


def run_failure(local_rank) -> None:
    raise RuntimeError("Test error")


def run_infinite(local_rank) -> None:
    while True:
        time.sleep(1)


class MpProcessContextTest(unittest.TestCase):
    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_success(self):
        nprocs = 4
        mult = 2
        params = [MpParameters(fn=run_compute, args=(mult,))] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        ret_vals = proc_context.wait()
        while not ret_vals:
            ret_vals = proc_context.wait()
        self.assertEqual(4, len(ret_vals))
        for local_rank, ret_val in ret_vals.items():
            self.assertEqual(mult * local_rank, ret_val)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_success_no_return_func(self):
        nprocs = 4
        params = [MpParameters(fn=run_dummy, args=())] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        ret_vals = proc_context.wait()
        while not ret_vals:
            ret_vals = proc_context.wait()
        self.assertEqual(4, len(ret_vals))
        for ret_val in ret_vals.values():
            self.assertEqual(None, ret_val)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_huge_output(self):
        # python multiprocessing.queue module uses pipes and actually PipedQueues
        # This means that if a single object is greater than a pipe size
        # the writer process will block until reader process will start
        # reading the pipe.
        # This test makes a worker fn to return huge output, around ~10 MB
        nprocs = 4
        size = 200000
        params = [MpParameters(fn=fill_dict, args=(size,))] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        ret_vals = proc_context.wait()
        while not ret_vals:
            ret_vals = proc_context.wait()
        self.assertEqual(4, len(ret_vals))
        for ret_val in ret_vals.values():
            self.assertEqual(size, len(ret_val))

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_failure(self):
        nprocs = 4
        params = [MpParameters(fn=run_failure, args=())] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        with self.assertRaises(ProcessRaisedException):
            ret_vals = proc_context.wait()
            while not ret_vals:
                ret_vals = proc_context.wait()

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_termination(self):
        nprocs = 5
        params = [MpParameters(fn=run_infinite, args=())] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        proc_context.terminate()
        # Processes should terminate with SIGTERM
        with self.assertRaises(Exception):
            proc_context.wait()

    def test_wait_busy_loop(self):
        nprocs = 2
        wait_time = 10  # seconds
        params = [MpParameters(fn=run_with_wait, args=(wait_time,))] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        self.assertIsNone(proc_context.wait(1))
        while proc_context.wait(1) is None:
            pass

    def test_wrap_fn(self):
        nprocs = 2
        start_method = "spawn"
        out_queues: Dict[int, mp.SimpleQueue] = {
            i: mp.get_context(start_method).SimpleQueue() for i in range(0, nprocs)
        }
        params = [MpParameters(fn=run_compute, args=(1,))] * nprocs
        for idx in range(nprocs):
            _wrap(idx, params, out_queues)
        for idx, out_q in out_queues.items():
            self.assertFalse(out_q.empty(), "out queue should not be empty")
            self.assertEqual(idx, out_q.get())
