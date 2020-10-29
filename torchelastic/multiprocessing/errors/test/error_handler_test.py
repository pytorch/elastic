#!/usr/bin/env python3
import json
import os
import shutil
import tempfile
import unittest

from torchelastic.multiprocessing.errors.error_handler import (
    ErrorHandler,
    ProcessException,
)


def raise_exception():
    raise RuntimeError("test error")


class ErrorHandlerTest(unittest.TestCase):
    def test_record_exception(self):
        test_dir = tempfile.mkdtemp()
        error_handler = ErrorHandler(test_dir)
        try:
            raise_exception()
        except Exception as e:
            error_handler.record_exception(e)
        error_file = error_handler._get_error_reply_file(os.getpid())
        self.assertTrue(os.path.exists(error_file))
        with open(error_file, "r") as f:
            data = json.load(f)
        self.assertTrue("RuntimeError: test error" in data["message"])
        shutil.rmtree(test_dir)

    def test_try_raise_exception(self):

        test_dir = tempfile.mkdtemp()
        error_handler = ErrorHandler(test_dir)
        try:
            raise_exception()
        except Exception as e:
            error_handler.record_exception(e)

        with self.assertRaisesRegex(
            ProcessException, "RuntimeError: test error"
        ) as context:
            error_handler.try_raise_exception(os.getpid())
        e = context.exception
        self.assertEqual(os.getpid(), e.pid)

        shutil.rmtree(test_dir)

    def test_try_raise_exception_no_file(self):

        test_dir = tempfile.mkdtemp()
        error_handler = ErrorHandler(test_dir)
        exit_code = 1
        child_pid = 19
        with self.assertRaisesRegex(
            ProcessException,
            f"Process {child_pid} terminated with exit code {exit_code}",
        ) as context:
            error_handler.try_raise_exception(child_pid, exit_code)
        e = context.exception
        self.assertEqual(child_pid, e.pid)

        shutil.rmtree(test_dir)
