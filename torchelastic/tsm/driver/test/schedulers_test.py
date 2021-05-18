#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchelastic.tsm.driver.schedulers import get_schedulers
from torchx.schedulers.local_scheduler import LocalScheduler


class SchedulersTest(unittest.TestCase):
    def test_get_local_schedulers(self):
        schedulers = get_schedulers(session_name="test_session")
        self.assertTrue(isinstance(schedulers["local"], LocalScheduler))
        self.assertTrue(isinstance(schedulers["default"], LocalScheduler))

        self.assertEquals("test_session", schedulers["local"].session_name)
        self.assertEquals("test_session", schedulers["default"].session_name)
