#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import tempfile
import unittest

from torchelastic.multiprocessing.errors import ProcessException
from torchelastic.multiprocessing.sp import (
    SubprocessParameters,
    _resolve_std_stream,
    start_processes,
)


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class SubprocessContextTest(unittest.TestCase):
    def _check_file(self, file_path, string) -> bool:
        with open(file_path, "r") as f:
            for line in f.readlines():
                if string in line:
                    return True
        return False

    def test_run_success(self):
        test_dir = tempfile.mkdtemp()
        nprocs = 4
        params_list = [
            SubprocessParameters(
                args=[path("bin/test_script.py")],
                stdout=test_dir,
                stderr=test_dir,
            )
        ] * nprocs
        proc_context = start_processes(params_list)
        completed_processes = proc_context.wait()
        for rank in range(len(completed_processes.values())):
            self.assertTrue(
                self._check_file(f"{test_dir}/{rank}/stdout.log", "Success")
            )
        shutil.rmtree(test_dir)

    def test_run_fail_group(self):
        test_dir = tempfile.mkdtemp()
        nprocs = 4
        params_list = [
            SubprocessParameters(
                args=[path("bin/test_script.py"), "--fail"],
                stdout=test_dir,
                stderr=test_dir,
            )
        ] * nprocs
        proc_context = start_processes(params_list)
        with self.assertRaises(ProcessException) as context:
            proc_context.wait()
        failed_proc_rank = 0
        for idx, proc in enumerate(proc_context.processes):
            if context.exception.pid == proc.pid:
                failed_proc_rank = idx
        self._check_file(
            f"{test_dir}/{failed_proc_rank}/stderr.log",
            "raising exception since --fail flag was set",
        )
        shutil.rmtree(test_dir)

    def test_run_async_success(self):
        nprocs = 4
        params_list = [
            SubprocessParameters(args=[path("bin/test_script.py"), "--wait", "5"])
        ] * nprocs
        proc_context = start_processes(params_list)
        self.assertIsNone(proc_context.wait(1))
        proc_context.wait(5)
        self.assertFalse(proc_context._any_alive())

    def test_run_different_timing(self):
        test_dir = tempfile.mkdtemp()
        params_list = [
            SubprocessParameters(
                args=[path("bin/test_script.py"), "--wait", "2", "--fail"],
                stdout=test_dir,
                stderr=test_dir,
            ),
            SubprocessParameters(
                args=[path("bin/test_script.py"), "--wait", "1"],
                stdout=test_dir,
                stderr=test_dir,
            ),
            SubprocessParameters(
                args=[path("bin/test_script.py"), "--wait", "5"],
                stdout=test_dir,
                stderr=test_dir,
            ),
        ]
        proc_context = start_processes(
            params_list,
        )
        with self.assertRaises(ProcessException) as context:
            proc_context.wait(timeout=3)

        failed_rank = 0
        for rank, proc in enumerate(proc_context.processes):
            if context.exception.pid == proc.pid:
                failed_rank = rank

        self._check_file(
            f"{test_dir}/{failed_rank}/stderr.log",
            "raising exception since --fail flag was set",
        )
        shutil.rmtree(test_dir)

    def _read_file(self, path):
        with open(path, "r") as file:
            return file.read().replace("\n", "")

    def _get_params(
        self,
        args,
        stdout=None,
        stderr=None,
    ) -> SubprocessParameters:
        return SubprocessParameters(args=args, stdout=stdout, stderr=stderr)

    def test_run_stream_redirect_to_file(self):
        test_dir = tempfile.mkdtemp()
        params_list = [
            self._get_params(
                [path("bin/test_script.py")],
                stdout=test_dir,
                stderr=test_dir,
            ),
            self._get_params(
                [path("bin/test_script.py"), "--fail"],
                stdout=test_dir,
                stderr=test_dir,
            ),
        ]
        proc_context = start_processes(params_list)
        with self.assertRaises(ProcessException):
            proc_context.wait()
        self.assertTrue(os.path.exists(f"{test_dir}/0/stdout.log"))
        self.assertTrue(os.path.exists(f"{test_dir}/0/stderr.log"))
        self.assertTrue(os.path.exists(f"{test_dir}/1/stdout.log"))
        self.assertTrue(os.path.exists(f"{test_dir}/1/stderr.log"))
        self.assertEqual("Success", self._read_file(f"{test_dir}/0/stdout.log"))
        self.assertTrue(
            "raising exception since --fail flag was set"
            in self._read_file(f"{test_dir}/1/stderr.log"),
        )
        shutil.rmtree(test_dir)

    def test_std_different_dest(self):
        test_dir = tempfile.mkdtemp()
        params_list = [
            self._get_params(
                [path("bin/test_script.py")],
                stdout=test_dir,
            ),
            self._get_params(
                [path("bin/test_script.py"), "--fail"],
                stdout=test_dir,
            ),
        ]
        params_list[1].args.append("--fail")
        proc_context = start_processes(params_list)
        with self.assertRaises(ProcessException) as context:
            proc_context.wait()
        self.assertTrue(os.path.exists(f"{test_dir}/0/stdout.log"))
        self.assertTrue(os.path.exists(f"{test_dir}/1/stdout.log"))
        self.assertEqual("Success", self._read_file(f"{test_dir}/0/stdout.log"))
        shutil.rmtree(test_dir)

    def test_wait_no_timeout(self):
        nprocs = 2
        params_list = [
            SubprocessParameters(
                args=[path("bin/test_script.py"), "--wait", "5"],
            )
        ] * nprocs
        proc_context = start_processes(params_list)

        self.assertEqual(None, proc_context.wait(1))
        self.assertEqual(None, proc_context.wait(1))
        proc_context.wait()

    def test_terminate_proc(self):
        nprocs = 2
        params_list = [
            SubprocessParameters(
                args=[path("bin/test_script.py"), "--run", "10"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        ] * nprocs
        proc_context = start_processes(params_list)
        proc_context.terminate()
        self.assertFalse(proc_context._any_alive())

    def test_resolve_std_stream(self):
        self.assertIsNone(_resolve_std_stream(None, "err", 0))
        self.assertEqual(0, _resolve_std_stream(0, "err", 0))
        self.assertEqual(-1, _resolve_std_stream(-1, "err", 0))
        std_dir = test_dir = tempfile.mkdtemp()
        out_std_dest = f"{std_dir}/0/err.log"
        std_stream_dest = _resolve_std_stream(std_dir, "err", 0)
        self.assertEqual(out_std_dest, std_stream_dest.name)
        shutil.rmtree(test_dir)
