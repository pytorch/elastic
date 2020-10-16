#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import Mock, patch

from torchelastic.multiprocessing.error_reporter.api import exec_fn, get_error, record


def sum_func(arg1: int, arg2: int) -> int:
    return arg1 + arg2


@record
def raise_excepiton() -> str:
    raise Exception("test exception")


class ErrorReporterApiTest(unittest.TestCase):
    @patch("torchelastic.multiprocessing.error_reporter.api.get_error_handler")
    def test_exec_fn(self, get_error_handler_mock):
        error_handler_mock = Mock()
        get_error_handler_mock.return_value = error_handler_mock
        res = exec_fn(sum_func, args=(1, 2))
        self.assertEqual(3, res)
        error_handler_mock.configure.assert_called_once()

    @patch("torchelastic.multiprocessing.error_reporter.api.get_error_handler")
    def test_get_error(self, get_error_handler_mock):
        error_handler_mock = Mock()
        get_error_handler_mock.return_value = error_handler_mock
        get_error(1234)
        error_handler_mock.get_process_error.assert_called_once()

    def test_record(self):
        mock_handler = Mock()
        with patch(
            "torchelastic.multiprocessing.error_reporter.api.get_error_handler",
            return_value=mock_handler,
        ):
            try:
                raise_excepiton()
            except Exception:
                mock_handler.record_exception.assert_called_once()
