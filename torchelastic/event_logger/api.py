#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time


class EventLogHandler(abc.ABC):
    @abc.abstractmethod
    def log_event(self, event_name, message):
        pass


class ConsoleEventLogHandler(EventLogHandler):
    def log_event(self, event_name, message):
        print("[{}][{}]: {}={}".format(time.time(), event_name, message))


class NullEventLogHandler(EventLogHandler):
    def log_event(self, event_name, message):
        pass


_event_logger_map = {}
_default_event_logger_handler = NullEventLogHandler()


def configure(handler, group=None):
    if group is None:
        global _default_event_logger_handler
        _default_event_logger_handler = handler
    else:
        _event_logger_map[group] = handler


def get_event_logger(group=None):
    return (
        _event_logger_map[group]
        if group in _event_logger_map
        else _default_event_logger_handler
    )
