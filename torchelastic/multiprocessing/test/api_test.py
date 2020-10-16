#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import unittest

from torchelastic.multiprocessing import (
    Params,
    ProcessGroupException,
    TerminationBehavior,
    run,
    run_async,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class ParamsTest(unittest.TestCase):
    def test_copy(self):
        params = Params(
            args=[1, 2, 3, "test"], stdout=1, stderr="string", test_arg="test_value"
        )
        cloned_params = params.copy_obj()
        self.assertTrue(params != cloned_params)
        self.assertListEqual([1, 2, 3, "test"], cloned_params.args)
        self.assertEqual(1, cloned_params.stdout)
        self.assertEqual("string", cloned_params.stderr)
        self.assertEqual("test_value", cloned_params.kwargs["test_arg"])


class ApiTest(unittest.TestCase):
    def _check_stream(self, stream, string) -> bool:
        for line in stream:
            if string in line.decode():
                return True
        return False

    def test_run_success(self):
        nprocs = 4
        params_list = Params(
            args=[path("bin/test_script.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).replicate(nprocs)
        completed_processes = run(params_list)
        for res in completed_processes:
            self.assertTrue(self._check_stream(res.stdout, "Success"))

    def _check_exception_stderr(
        self, group_exception: ProcessGroupException, expected_failed_processes: int
    ) -> None:
        self.assertEqual(expected_failed_processes, len(group_exception.get_errors()))
        for exception in group_exception.get_errors().values():
            self._check_stream(
                exception.stderr, "raising exception since --fail flag was set"
            )

    def test_run_fail_single(self):
        nprocs = 2
        params_list = Params(
            args=[path("bin/test_script.py"), "--fail"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).replicate(nprocs)
        params_list[0].args += ["--wait", "5"]
        with self.assertRaises(ProcessGroupException) as context:
            run(params_list, termination=TerminationBehavior.SINGLE)
        self._check_exception_stderr(context.exception, 1)

    def test_run_fail_group(self):
        nprocs = 4
        params_list = Params(
            args=[path("bin/test_script.py"), "--fail"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).replicate(nprocs)
        with self.assertRaises(ProcessGroupException) as context:
            run(params_list, termination=TerminationBehavior.GROUP)
        self._check_exception_stderr(context.exception, 4)

    def test_run_async_success(self):
        nprocs = 4
        params_list = Params(
            args=[path("bin/test_script.py"), "--wait", "5"]
        ).replicate(nprocs)
        proc_context = run_async(params_list)
        self.assertIsNone(proc_context.check())
        proc_context.wait(5)
        self.assertFalse(proc_context.any_alive())

    def test_run_different_timing(self):
        params_list = [
            Params(
                args=[path("bin/test_script.py"), "--wait", "2", "--fail"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
            Params(
                args=[path("bin/test_script.py"), "--wait", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
            Params(
                args=[path("bin/test_script.py"), "--wait", "5"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
        ]
        with self.assertRaises(ProcessGroupException) as context:
            run(
                params_list,
                timeout=3,
                termination_timeout=6,
                termination=TerminationBehavior.GROUP,
            )
        self._check_exception_stderr(context.exception, 1)
