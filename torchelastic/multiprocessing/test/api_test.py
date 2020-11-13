#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

from torchelastic.multiprocessing import (
    MpParameters,
    SubprocessParameters,
    start_processes,
    start_subprocesses,
)


def dummy_fn():
    pass


class InitTest(unittest.TestCase):
    @patch("torchelastic.multiprocessing.mp.start_processes")
    def test_invoke_mp(self, mp_mock):
        params = [MpParameters(fn=dummy_fn, args=())] * 4
        start_processes(params, start_method="fork")
        mp_mock.assert_called_once_with(params, "fork", 0)

    def test_invoke_mp_no_params(self):
        with self.assertRaises(ValueError):
            start_processes([], start_method="fork")

    def test_invoke_sp_no_params(self):
        with self.assertRaises(ValueError):
            start_subprocesses([])

    @patch("torchelastic.multiprocessing.sp.start_processes")
    def test_invoke_sp(self, sp_mock):
        params = [SubprocessParameters(args=[], test_key="test_value")] * 4
        start_subprocesses(params)
        sp_mock.assert_called_once_with(params, 0)
