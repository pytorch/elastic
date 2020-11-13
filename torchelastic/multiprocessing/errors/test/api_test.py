#!/usr/bin/env python3
import os
import unittest

from torchelastic.multiprocessing.errors.api import (
    ProcessFailure,
    _process_error_handler,
    get_error_file,
    get_failed_result,
    record,
)


@record
def raise_exception_non_root():
    raise RuntimeError("test error")


@record()
def raise_exception_root():
    raise RuntimeError("test error")


class ApiTest(unittest.TestCase):
    def test_record_non_root(self):
        with self.assertRaises(RuntimeError):
            raise_exception_non_root()
        error_file_path = _process_error_handler.error_file
        self.assertTrue(os.path.exists(error_file_path))

    def test_record_root(self):
        with self.assertRaises(RuntimeError):
            raise_exception_root()
        error_file_path = _process_error_handler.error_file
        self.assertTrue(os.path.exists(error_file_path))

    def test_get_failed_result(self):
        result = get_failed_result(0, 1, 2)
        self.assertTrue(isinstance(result, ProcessFailure))

    def test_get_error_file(self):
        _process_error_handler.configure()
        parent_error_file = _process_error_handler.error_file
        error_file = get_error_file(42)
        expected_error_file = os.path.join(
            os.path.dirname(parent_error_file), str(42), "error.log_0"
        )
        self.assertEqual(expected_error_file, error_file)
        _process_error_handler.cleanup()

    def test_get_error_file_with_run_id(self):
        _process_error_handler.configure()
        parent_error_file = _process_error_handler.error_file
        error_file = get_error_file(42, run_id=18)
        expected_error_file = os.path.join(
            os.path.dirname(parent_error_file), str(42), "error.log_18"
        )
        self.assertEqual(expected_error_file, error_file)
        _process_error_handler.cleanup()
