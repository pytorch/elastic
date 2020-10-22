#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest

from torch.multiprocessing.spawn import ProcessRaisedException
from torchelastic.multiprocessing.mp import MpParameters, start_processes
from torchelastic.test.test_utils import is_tsan


def run_compute(local_rank, mult=1) -> int:
    time.sleep(1)
    return local_rank * mult


def run_dummy(local_rank) -> None:
    time.sleep(1)


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
