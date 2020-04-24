#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
from typing import Any, Dict


class Event:
    """
    The class represents the generic event that occurs during the torchelastic
    job execution. The event can be any kind of meaningful action.

    Args:
        name: event name.
        type: type of event, determines what handler will process the event.
        metadata: additional data that is associated with the event.
        timestamp: timestamp in milliseconds when event occured.
    """

    __slots__ = ["name", "type", "metadata", "timestamp"]

    def __init__(
        self,
        name: str,
        type: str = "torchelastic",
        # Cannot use optional due to https://github.com/facebook/pyre-check/issues/197
        # pyre-fixme[9]: metadata has type `Dict[str, Any]`; used as `None`.
        metadata: Dict[str, Any] = None,
        timestamp: int = 0,
    ):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.timestamp = timestamp


class EventHandler(abc.ABC):
    """
    Event handler interface is responsible for capturing and recording events
    to the different destinations, e.g. stdout, file, etc.
    """

    @abc.abstractmethod
    def record(self, event: Event):
        raise NotImplementedError()


class ConsoleEventHandler(EventHandler):
    """
    The event handler that prints the captured events to the console.
    """

    def record(self, event: Event):
        print(f"[{event.timestamp}] {event.type}:{event.name}")


class NullEventHandler(EventHandler):
    """
    The default event handler that is a no-op.
    """

    def record(self, event: Event):
        pass


_events_map: Dict[str, EventHandler] = {}
_default_event_handler: EventHandler = NullEventHandler()


def _getEventHandler(event_type: str):
    if event_type in _events_map:
        return _events_map[event_type]
    return _default_event_handler


def configure(handler: EventHandler, event_type: str = ""):
    r"""
    Sets the relation between event_type and handler. After the relation is set
    all events that have type `event_type` will be processed by the handler.

    Args:
        handler: Event handler that will process the events.
        event_type: Type of event.
    """
    if not event_type:
        global _default_event_handler
        _default_event_handler = handler
    else:
        _events_map[event_type] = handler


def record_event(event: Event):
    r"""
    Records the event to the destination configured by the ``event.type`` parameter.

    Args:
        event (Event): Event to be recorded
    """
    if event.metadata is None:
        event.metadata = {}
    _getEventHandler(event.type).record(event)


def record(
    event_name: str,
    event_type: str = "torchelastic",
    # Cannot use optional due to https://github.com/facebook/pyre-check/issues/197
    # pyre-fixme[9]: metadata has type `Dict[str, Any]`; used as `None`.
    metadata: Dict[str, Any] = None,
):
    r"""
    Constructs and  records the event to the destination
    configured by the event.type parameter.

    Args:
        event_name: The name of the event
        event_type: The type of the event. Is used to determine what handler
            will process the event
        metadata: Arbitrary data that is associated with the event.
    """

    if metadata is None:
        metadata = {}
    event = Event(
        event_name, event_type, metadata, timestamp=int(round(time.time() * 1000))
    )
    record_event(event)
