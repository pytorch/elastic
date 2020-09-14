#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchelastic.tsm.driver.schedulers import get_scheduler


class SchedulersTest(unittest.TestCase):
    def test_get_local_schedulers(self):
        scheduler = get_scheduler("local", cache_size=250)
        self.assertIsNotNone(scheduler)
        self.assertEquals(250, scheduler._cache_size)

    def test_get_unknown_scheduler(self):
        with self.assertRaises(ValueError):
            get_scheduler("unknown_scheduler", cache_size=250)
