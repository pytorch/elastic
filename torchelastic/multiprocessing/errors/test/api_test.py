#!/usr/bin/env python3
import os
import unittest

from torchelastic.multiprocessing.errors import ProcessException
from torchelastic.multiprocessing.errors.api import (
    _process_error_handler,
    record,
    try_raise_exception,
)


@record
def raise_exception_non_root():
    raise RuntimeError("test error")


@record(root_process=True)
def raise_exception_root():
    raise RuntimeError("test error")


class ApiTest(unittest.TestCase):
    def test_record_non_root(self):
        with self.assertRaises(RuntimeError):
            raise_exception_non_root()
        error_file_path = _process_error_handler._get_error_file_path(os.getpid())
        self.assertTrue(os.path.exists(error_file_path))

    def test_record_root(self):
        with self.assertRaises(RuntimeError):
            raise_exception_root()
        error_file_path = _process_error_handler._get_error_file_path(os.getpid())
        self.assertTrue(os.path.exists(error_file_path))

    def test_try_raise_exception(self):
        with self.assertRaises(RuntimeError):
            raise_exception_non_root()
        with self.assertRaisesRegex(ProcessException, "test error") as context:
            try_raise_exception(os.getpid())
        e = context.exception
        self.assertEqual(os.getpid(), e.pid)
