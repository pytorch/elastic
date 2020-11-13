#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import os
import shutil
import tempfile
import time
import unittest
from typing import Dict

import torch.multiprocessing as mp
from torchelastic.multiprocessing.api import ProcessGroupResult
from torchelastic.multiprocessing.errors.api import _process_error_handler
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


def run_failure_signal(local_rank) -> None:
    ctypes.string_at(0)


class MpProcessContextTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_success(self):
        nprocs = 4
        mult = 2
        params = [MpParameters(fn=run_compute, args=(mult,))] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        proc_group_result = self._get_result(proc_context)
        ret_vals = proc_group_result.return_values
        self.assertEqual(4, len(ret_vals))
        for local_rank, ret_val in ret_vals.items():
            self.assertEqual(mult * local_rank, ret_val)

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_success_no_return_func(self):
        nprocs = 4
        params = [MpParameters(fn=run_dummy, args=())] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        proc_group_result = self._get_result(proc_context)
        ret_vals = proc_group_result.return_values
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
        proc_group_result = self._get_result(proc_context)
        ret_vals = proc_group_result.return_values
        self.assertEqual(4, len(ret_vals))
        for ret_val in ret_vals.values():
            self.assertEqual(size, len(ret_val))

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_run_failure(self):
        os.environ["TORCHELASTIC_ERROR_FILE"] = f"{self.test_dir}/error.log"
        _process_error_handler.configure()
        nprocs = 4
        params = [MpParameters(fn=run_failure, args=())] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        proc_group_result = self._get_result(proc_context)
        failed_result = proc_group_result.failure
        self.assertTrue(os.path.exists(failed_result.error_file))
        with open(failed_result.error_file, "r") as f:
            data = f.read().replace("\n", "")
        self.assertTrue("RuntimeError: Test error" in data)
        _process_error_handler.cleanup()

    def _get_result(self, proc_context) -> ProcessGroupResult:
        proc_group_result = proc_context.wait()
        while not proc_group_result:
            proc_group_result = proc_context.wait()
        return proc_group_result

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_failure_signal(self):
        os.environ["TORCHELASTIC_ERROR_FILE"] = f"{self.test_dir}/error.log"
        _process_error_handler.configure()
        nprocs = 5
        params = [MpParameters(fn=run_failure_signal, args=())] * nprocs
        proc_context = start_processes(params, start_method="spawn")
        # Processes should terminate with SIGSEGV
        proc_group_result = proc_context.wait()
        failure = proc_group_result.failure
        self.assertTrue(os.path.exists(failure.error_file))
        self.assertEqual("SIGSEGV", failure.get_signal_name())
        with open(failure.error_file, "r") as f:
            data = f.read().replace("\n", "")
        self.assertTrue("string_at" in data)
        _process_error_handler.cleanup()

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
        error_files = ["error.log" for _ in range(0, nprocs)]
        params = [MpParameters(fn=run_compute, args=(1,))] * nprocs
        for idx in range(nprocs):
            _wrap(idx, error_files, params, out_queues)
        for idx, out_q in out_queues.items():
            self.assertFalse(out_q.empty(), "out queue should not be empty")
            self.assertEqual(idx, out_q.get())
