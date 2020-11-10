#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torchelastic.utils.logging as logging


log = logging.get_logger()


class LoggingTest(unittest.TestCase):
    def test_logger_name(self):
        self.assertEqual("torchelastic.utils.test.logging_test", log.name)

    def test_derive_module_name(self):
        module_name = logging._derive_module_name(depth=1)
        self.assertEqual("torchelastic.utils.test.logging_test", module_name)
