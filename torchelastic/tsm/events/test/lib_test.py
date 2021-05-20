#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs
import json
import logging
import unittest
from unittest.mock import patch, MagicMock

from torchelastic.tsm.events import (
    _get_or_create_logger,
    SourceType,
    TsmEvent,
    log_event,
)


class TsmEventLibTest(unittest.TestCase):
    def assert_event(self, actual_event: TsmEvent, expected_event: TsmEvent):
        self.assertEqual(actual_event.session, expected_event.session)
        self.assertEqual(actual_event.scheduler, expected_event.scheduler)
        self.assertEqual(actual_event.api, expected_event.api)
        self.assertEqual(actual_event.app_id, expected_event.app_id)
        self.assertEqual(actual_event.runcfg, expected_event.runcfg)
        self.assertEqual(actual_event.source, expected_event.source)

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
        self.assertEqual(SourceType.UNKNOWN, event.source)

    def test_event_deser(self):
        event = TsmEvent(
            session="test_session",
            scheduler="test_scheduler",
            api="test_api",
            source=SourceType.EXTERNAL,
        )
        json_event = event.serialize()
        deser_event = TsmEvent.deserialize(json_event)
        self.assert_event(event, deser_event)


@patch("torchelastic.tsm.events.record")
class LogEventTest(unittest.TestCase):
    def assert_tsm_event(self, expected: TsmEvent, actual: TsmEvent) -> None:
        self.assertEqual(expected.session, actual.session)
        self.assertEqual(expected.app_id, actual.app_id)
        self.assertEqual(expected.api, actual.api)
        self.assertEqual(expected.source, actual.source)

    def test_create_context(self, _) -> None:
        cfg = json.dumps({"test_key": "test_value"})
        context = log_event("test_call", "local", "test_app_id", cfg)
        expected_tsm_event = TsmEvent(
            "test_app_id", "local", "test_call", "test_app_id", cfg
        )
        self.assert_tsm_event(expected_tsm_event, context._tsm_event)

    def test_record_event(self, record_mock: MagicMock) -> None:
        cfg = json.dumps({"test_key": "test_value"})
        expected_tsm_event = TsmEvent(
            "test_app_id", "local", "test_call", "test_app_id", cfg
        )
        with log_event("test_call", "local", "test_app_id", cfg) as ctx:
            pass
        self.assert_tsm_event(expected_tsm_event, ctx._tsm_event)

    def test_record_event_with_exception(self, record_mock: MagicMock) -> None:
        cfg = json.dumps({"test_key": "test_value"})
        with self.assertRaises(RuntimeError):
            with log_event("test_call", "local", "test_app_id", cfg) as ctx:
                raise RuntimeError("test error")
        self.assertTrue("test error" in ctx._tsm_event.raw_exception)
