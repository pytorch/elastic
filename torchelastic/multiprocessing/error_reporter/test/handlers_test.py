#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchelastic.multiprocessing.error_reporter.handlers import get_signal_handler
from torchelastic.multiprocessing.error_reporter.signal_handler import (
    LocalSignalHandler,
)


class SignalHandlerFactoryTest(unittest.TestCase):
    def test_get_local_signal_handler(self):
        signal_handler = get_signal_handler()
        self.assertTrue(isinstance(signal_handler, LocalSignalHandler))
