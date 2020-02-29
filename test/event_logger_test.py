#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchelastic.event_logger import EventLogHandler, configure, get_event_logger


class MockEventLogHandler(EventLogHandler):
    def __init__(self, event_log_map):
        self.event_log_map = event_log_map

    def log_event(self, event_name, message):
        self.event_log_map[event_name] = message


class EventLoggerTest(unittest.TestCase):
    def test_default_event_logger(self):
        # default is the null handler, simply check that
        # adding to a un-configured event logger group does not throw
        event_logger = get_event_logger()
        event_logger.log_event("a", "b")
        # nothing to assert, counter goes to a black hole sink

    def test_set_event_logger_handler(self):
        event_log_map = {}
        configure(MockEventLogHandler(event_log_map))
        event_logger = get_event_logger()
        event_logger.log_event("a", "b")
        self.assertEqual("b", event_log_map["a"])

        my_event_log_map = {}
        configure(MockEventLogHandler(my_event_log_map), "my")
        event_logger = get_event_logger("my")
        event_logger.log_event("a", "c")
        self.assertEqual("c", my_event_log_map["a"])
        # test not mix
        self.assertEqual("b", event_log_map["a"])
