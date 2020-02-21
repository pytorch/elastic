#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing as mp
import signal
import time
import unittest

import torch.multiprocessing as torch_mp
import torchelastic.timer as timer
from test_utils import is_asan


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
)


def _happy_function(rank, mp_queue):
    timer.configure(timer.LocalTimerClient(mp_queue))
    with timer.expires(after=1):
        time.sleep(0.5)


def _stuck_function(rank, mp_queue):
    timer.configure(timer.LocalTimerClient(mp_queue))
    with timer.expires(after=1):
        time.sleep(5)


class LocalTimerExample(unittest.TestCase):
    """
    Demonstrates how to use LocalTimerServer and LocalTimerClient
    to enforce expiration of code-blocks.

    Since torch multiprocessing's ``start_process`` method currently
    does not take the multiprocessing context as parameter argument
    there is no way to create the mp.Queue in the correct
    context BEFORE spawning child processes. Once the ``start_process``
    API is changed in torch, then re-enable ``test_torch_mp_example``
    unittest. As of now this will SIGSEGV.
    """

    @unittest.skip("re-enable when torch_mp.spawn() takes mp context as param")
    def test_torch_mp_example(self):
        # in practice set the max_interval to a larger value (e.g. 60 seconds)
        ctx = mp.get_context("spawn")
        mp_queue = ctx.Queue()
        server = timer.LocalTimerServer(mp_queue, max_interval=0.01)
        server.start()

        world_size = 8
        # all processes should complete successfully
        # since start_process does NOT take context as parameter argument yet
        # this method WILL FAIL (hence the test is disabled)
        torch_mp.start_process(
            fn=_happy_function, args=(mp_queue,), nprocs=world_size, context=ctx
        )

        with self.assertRaises(Exception):
            # torch.multiprocessing.spawn kills all sub-procs
            # if one of them gets killed
            torch_mp.start_process(
                fn=_stuck_function, args=(mp_queue,), nprocs=world_size, context=ctx
            )

        server.stop()

    @unittest.skipIf(is_asan(), "test is asan incompatible")
    def test_example_start_method_spawn(self):
        self._run_example_with(start_method="spawn")

    @unittest.skipIf(is_asan(), "test is asan incompatible")
    def test_example_start_method_forkserver(self):
        self._run_example_with(start_method="forkserver")

    def test_example_start_method_fork(self):
        self._run_example_with(start_method="fork")

    def _run_example_with(self, start_method):
        spawn_ctx = mp.get_context(start_method)
        mp_queue = spawn_ctx.Queue()
        server = timer.LocalTimerServer(mp_queue, max_interval=0.01)
        server.start()

        world_size = 8
        processes = []
        for i in range(0, world_size):
            if i % 2 == 0:
                p = spawn_ctx.Process(target=_stuck_function, args=(i, mp_queue))
            else:
                p = spawn_ctx.Process(target=_happy_function, args=(i, mp_queue))
            p.start()
            processes.append(p)

        for i in range(0, world_size):
            p = processes[i]
            p.join()
            if i % 2 == 0:
                self.assertEqual(-signal.SIGKILL, p.exitcode)
            else:
                self.assertEqual(0, p.exitcode)

        server.stop()
