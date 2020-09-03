#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import random
import signal
import sys
import time
import unittest
from unittest.mock import patch

import torch.multiprocessing as torch_mp
from test_utils import is_asan_or_tsan
from torchelastic.multiprocessing.spawn import (
    WorkerExitedException,
    WorkerSignaledException,
    spawn,
)


def test_success_func(i, arg=None):
    if arg:
        arg.put(i)
    pass


def test_exception_single_func(i, arg):
    if i == arg:
        raise ValueError("legitimate exception from process %d" % i)
    time.sleep(1.0)


def test_exception_all_func(i):
    time.sleep(random.random() / 10)
    raise ValueError("legitimate exception from process %d" % i)


def test_terminate_signal_func(i):
    if i == 0:
        os.kill(os.getpid(), signal.SIGABRT)
    time.sleep(1.0)


def test_terminate_exit_func(i, arg):
    if i == 0:
        sys.exit(arg)
    time.sleep(1.0)


def test_success_first_then_exception_func(i, arg):
    if i == 0:
        return
    time.sleep(0.1)
    raise ValueError("legitimate exception")


class TorchElasticSpawnTest(unittest.TestCase):
    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_success(self):
        spawn(test_success_func, nprocs=2)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_success_non_blocking(self):
        spawn_context = spawn(test_success_func, nprocs=2, join=False)

        # After all processes (nproc=2) have joined it must return True
        spawn_context.join(timeout=None)
        spawn_context.join(timeout=None)
        self.assertTrue(spawn_context.join(timeout=None))

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_first_argument_index(self):
        mp = multiprocessing.get_context("spawn")
        queue = mp.SimpleQueue()
        spawn(test_success_func, (queue,), nprocs=2)
        self.assertEqual([0, 1], sorted([queue.get(), queue.get()]))

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_exception_single(self):
        nprocs = 2
        for i in range(nprocs):
            with self.assertRaisesRegex(
                Exception, "\nValueError: legitimate exception from process %d$" % i
            ):
                spawn(test_exception_single_func, (i,), nprocs=nprocs)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_exception_all(self):
        with self.assertRaisesRegex(
            Exception, "\nValueError: legitimate exception from process (0|1)$"
        ):
            spawn(test_exception_all_func, (), nprocs=2)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    @patch("torchelastic.multiprocessing.error_reporter.get_error")
    def test_terminate_signal(self, mock_get_error):
        subprocess_error_msg = (
            '*** Aborted at 1225095260 (unix time) try "date -d @1225095260" if you are using GNU date ***'
            "*** SIGABRT (@0x0) received by PID 17711 (TID 0x7f893090a6f0) from PID 0; stack trace: ***"
            "PC: @           0x412eb1 TestWaitingLogSink::send()"
            "    @     0x7f892fb417d0 (unknown)"
        )
        mock_get_error.return_value = subprocess_error_msg
        with self.assertRaises(WorkerSignaledException, msg=subprocess_error_msg) as cm:
            spawn(test_terminate_signal_func, (), nprocs=2)

        self.assertEqual("SIGABRT", cm.exception.signal_name)
        print(f"exception: {cm.exception}")
        self.assertEqual(0, cm.exception.error_index)
        self.assertEqual(subprocess_error_msg, str(cm.exception))

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    @patch("torchelastic.multiprocessing.error_reporter.get_error")
    def test_terminate_exit(self, mock_get_error):
        exitcode = 123
        subprocess_error_msg = "some_error_msg\n\n2333\ntrace:\n\n(fds)"
        mock_get_error.return_value = subprocess_error_msg
        with self.assertRaises(WorkerExitedException, msg=subprocess_error_msg) as cm:
            spawn(test_terminate_exit_func, (exitcode,), nprocs=2)
        self.assertEqual(exitcode, cm.exception.exit_code)
        self.assertEqual(0, cm.exception.error_index)
        self.assertEqual(subprocess_error_msg, str(cm.exception))

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_success_first_then_exception(self):
        exitcode = 123
        with self.assertRaisesRegex(Exception, "ValueError: legitimate exception"):
            spawn(test_success_first_then_exception_func, (exitcode,), nprocs=2)


"""
torch.multiprocessing.spawn test, make sure torchelastic.multiprocessing.spawn
does not interfere with torch.multiprocessing.spawn
"""


class TorchSpawnTest(unittest.TestCase):
    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_success(self):
        torch_mp.spawn(test_success_func, nprocs=2)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_success_non_blocking(self):
        spawn_context = torch_mp.spawn(test_success_func, nprocs=2, join=False)

        # After all processes (nproc=2) have joined it must return True
        spawn_context.join(timeout=None)
        spawn_context.join(timeout=None)
        self.assertTrue(spawn_context.join(timeout=None))

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_first_argument_index(self):
        mp = multiprocessing.get_context("spawn")
        queue = mp.SimpleQueue()
        torch_mp.spawn(test_success_func, (queue,), nprocs=2)
        self.assertEqual([0, 1], sorted([queue.get(), queue.get()]))

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_exception_single(self):
        nprocs = 2
        for i in range(nprocs):
            with self.assertRaisesRegex(
                Exception, "\nValueError: legitimate exception from process %d$" % i
            ):
                torch_mp.spawn(test_exception_single_func, (i,), nprocs=nprocs)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_exception_all(self):
        with self.assertRaisesRegex(
            Exception, "\nValueError: legitimate exception from process (0|1)$"
        ):
            torch_mp.spawn(test_exception_all_func, (), nprocs=2)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_terminate_signal(self):
        with self.assertRaisesRegex(
            Exception, r"process 0 terminated with signal SIGABRT"
        ):
            torch_mp.spawn(test_terminate_signal_func, (), nprocs=2)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_terminate_exit(self):
        exitcode = 123
        with self.assertRaisesRegex(
            Exception, r"process 0 terminated with exit code 123"
        ):
            torch_mp.spawn(test_terminate_exit_func, (exitcode,), nprocs=2)

    @unittest.skipIf(is_asan_or_tsan(), "test incompatible with asan or tsan")
    def test_success_first_then_exception(self):
        exitcode = 123
        with self.assertRaisesRegex(Exception, r"ValueError: legitimate exception"):
            torch_mp.spawn(
                test_success_first_then_exception_func, (exitcode,), nprocs=2
            )
