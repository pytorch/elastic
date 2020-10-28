#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchelastic.tsm.driver.local_scheduler import LocalScheduler
from torchelastic.tsm.driver.schedulers import get_schedulers


class SchedulersTest(unittest.TestCase):
    def test_get_local_schedulers(self):
        schedulers = get_schedulers()
        self.assertTrue(isinstance(schedulers["local"], LocalScheduler))
        self.assertTrue(isinstance(schedulers["default"], LocalScheduler))
