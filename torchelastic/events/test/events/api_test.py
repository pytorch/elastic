#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs
import unittest
import uuid

from torchelastic.events.api import Event, EventHandler, configure, record, record_event


class TestEventHandler(EventHandler):
    def __init__(self):
        self.events = []

    def record(self, event: Event):
        self.events.append(event)


class EventApiTest(unittest.TestCase):
    def assert_event(self, actual_event, expected_event):
        self.assertEqual(actual_event.name, expected_event.name)
        self.assertEqual(actual_event.type, expected_event.type)
        self.assertDictEqual(actual_event.metadata, expected_event.metadata)

    def test_configure_default_event_handler(self):
        event_handler = TestEventHandler()
        configure(event_handler)
        expected_event = Event("name", metadata={})
        record_event(expected_event)
        actual_event = event_handler.events[0]
        self.assert_event(actual_event, expected_event)

    def test_record(self):
        event_name = "test_event"
        event_handler = TestEventHandler()
        event_type = str(uuid.uuid4().int)
        configure(event_handler, event_type)
        metadata = {"test_key": 10, "test_key2": "test_value"}
        record(event_name, event_type, metadata)
        expected_event = Event(event_name, event_type, metadata)
        actual_event = event_handler.events[0]
        self.assert_event(actual_event, expected_event)

    def test_record_no_metadata(self):
        event_name = "test_event"
        event_handler = TestEventHandler()
        event_type = str(uuid.uuid4().int)
        configure(event_handler, event_type)
        record(event_name, event_type)
        expected_event = Event(event_name, event_type, {})
        actual_event = event_handler.events[0]
        self.assert_event(actual_event, expected_event)

    def test_record_event(self):
        event_type = str(uuid.uuid4().int)
        metadata = {"test_key": 10, "test_key2": "test_value"}
        expected_event = Event("test_event", event_type, metadata)
        event_handler = TestEventHandler()
        configure(event_handler, event_type)
        record_event(expected_event)
        actual_event = event_handler.events[0]
        self.assert_event(actual_event, expected_event)
