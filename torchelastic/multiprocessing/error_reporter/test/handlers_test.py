#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchelastic.multiprocessing.error_reporter.error_handler import LocalErrorHandler
from torchelastic.multiprocessing.error_reporter.handlers import get_error_handler


class SignalHandlerFactoryTest(unittest.TestCase):
    def test_get_local_error_handler(self):
        error_handler = get_error_handler()
        self.assertTrue(isinstance(error_handler, LocalErrorHandler))
