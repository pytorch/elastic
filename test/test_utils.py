#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import multiprocessing
import queue
import uuid


def is_asan():
    """Determines if the Python interpreter is running with ASAN"""
    return hasattr(ctypes.CDLL(""), "__asan_init")


def is_tsan():
    """Determines if the Python interpreter is running with TSAN"""
    return hasattr(ctypes.CDLL(""), "__tsan_init")


def is_asan_or_tsan():
    return is_asan() or is_tsan()


def _get_or_raise(qout, qerr):
    # Helper function that either returns the value on the out queue
    # or raises the exception on the error queue.
    while True:
        try:
            return qout.get(False, 0.001)
        except queue.Empty:
            pass
        try:
            raise qerr.get(False, 0.001)
        except queue.Empty:
            pass


class TestCommon:
    def setUp(self):
        self.processes = []

    def tearDown(self):
        for p in self.processes:
            p.terminate()

    def _generate_run_id(self):
        return str(uuid.uuid4().int)

    def _spawn(self, fn, *args):
        name = "process #%d" % len(self.processes)
        qin = multiprocessing.Queue()
        qout = multiprocessing.Queue()
        qerr = multiprocessing.Queue()
        qio = (qin, qout, qerr)
        args = qio + (fn,) + args
        process = multiprocessing.Process(target=self._run, name=name, args=args)
        process.start()
        self.processes.append(process)
        return qio

    def _wait_all_and_clean(self):
        for p in self.processes:
            p.join()
        self.processes = []

    def _run(self, qin, qout, qerr, fn, *args):
        try:
            qout.put(fn(qin, *args))
        except Exception as e:
            qerr.put(e)
