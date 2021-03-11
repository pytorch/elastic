#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs
import logging
import unittest
from unittest.mock import patch

from torchelastic.tsm.events import _get_or_create_logger, TsmEvent


class TsmEventLibTest(unittest.TestCase):
    def assert_event(self, actual_event: TsmEvent, expected_event: TsmEvent):
        self.assertEqual(actual_event.session, expected_event.session)
        self.assertEqual(actual_event.scheduler, expected_event.scheduler)
        self.assertEqual(actual_event.api, expected_event.api)
        self.assertEqual(actual_event.app_id, expected_event.app_id)
        self.assertEqual(actual_event.runcfg, expected_event.runcfg)

    @patch("torchelastic.tsm.events.get_logging_handler")
    def test_get_or_create_logger(self, logging_handler_mock):
        logging_handler_mock.return_value = logging.NullHandler()
        logger = _get_or_create_logger("test_destination")
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)

    def test_event_created(self):
        event = TsmEvent(
            session="test_session", scheduler="test_scheduler", api="test_api"
        )
        self.assertEqual("test_session", event.session)
        self.assertEqual("test_scheduler", event.scheduler)
        self.assertEqual("test_api", event.api)

    def test_event_deser(self):
        event = TsmEvent(
            session="test_session", scheduler="test_scheduler", api="test_api"
        )
        json_event = event.serialize()
        deser_event = TsmEvent.deserialize(json_event)
        self.assert_event(event, deser_event)
