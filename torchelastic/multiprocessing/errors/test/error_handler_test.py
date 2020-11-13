#!/usr/bin/env python3
import json
import os
import shutil
import tempfile
import unittest

from torchelastic.multiprocessing.errors.error_handler import (
    ErrorHandler,
    ProcessFailure,
)


def raise_exception():
    raise RuntimeError("test error")


class ErrorHandlerTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.environ.pop("TORCHELASTIC_ERROR_FILE", None)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_record_exception(self):
        error_handler = ErrorHandler()
        error_handler.error_file = os.path.join(self.test_dir, "error.log")
        try:
            raise_exception()
        except Exception as e:
            error_handler.record_exception(e)
        error_file = error_handler.error_file
        self.assertTrue(os.path.exists(error_file))
        with open(error_file, "r") as f:
            data = json.load(f)
        self.assertTrue("RuntimeError: test error" in data["message"])

    def test_get_failed_result(self):
        child_rank = 0
        # worker
        error_handler = ErrorHandler()
        error_handler.error_file = os.path.join(
            self.test_dir, str(child_rank), "error.log_0"
        )
        error_handler.configure()
        try:
            raise_exception()
        except Exception as e:
            error_handler.record_exception(e)

        # agent
        error_handler = ErrorHandler()
        error_handler.error_file = os.path.join(self.test_dir, "error.log")
        failed_result = error_handler.get_failed_result(child_rank, os.getpid(), 0)
        self.assertEqual(child_rank, failed_result.rank)
        self.assertEqual(os.getpid(), failed_result.pid)
        self.assertEqual(0, failed_result.exit_code)
        self.assertTrue(os.path.exists(failed_result.error_file))
        with open(failed_result.error_file, "r") as f:
            data = json.load(f)
        self.assertTrue("RuntimeError: test error" in data["message"])

    def test_get_failed_result_no_file(self):
        child_rank = 0
        # worker
        error_handler = ErrorHandler()
        error_handler.error_file = os.path.join(
            self.test_dir, str(child_rank), "error.log"
        )
        error_handler.configure()
        try:
            raise_exception()
        except Exception as e:
            error_handler.record_exception(e)

        # agent
        error_handler = ErrorHandler()
        error_handler.error_file = os.path.join(self.test_dir, "some_dir", "error.log")
        error_handler.configure()
        failed_result = error_handler.get_failed_result(child_rank, os.getpid(), 0)
        self.assertIsNone(failed_result.error_file)

    def test_process_failure_no_child_file(self):
        error_handler = ErrorHandler()
        error_handler.configure()

        failre = ProcessFailure("non_existent", 0, 0, 0, 0)

        error_handler.process_failure(failre)

    def test_process_failure(self):
        error_handler = ErrorHandler()
        error_handler.configure()
        error_handler.error_file = f"{self.test_dir}/error.log"
        child_error_file = f"{self.test_dir}/child_error.log"
        data = {"message": "test error"}
        with open(child_error_file, "w") as f:
            json.dump(data, f)
        failre = ProcessFailure(child_error_file, 0, 0, 0, 0)
        error_handler.process_failure(failre)
        self.assertTrue(os.path.exists(error_handler.error_file))

    def test_cleanup(self):
        error_handler = ErrorHandler()
        error_handler.configure()
        self.assertTrue(os.path.exists(error_handler.temp_dir))
        error_handler.cleanup()
        self.assertFalse(os.path.exists(error_handler.temp_dir))
