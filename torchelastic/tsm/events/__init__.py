#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module contains events processing mechanisms that are integrated with the standard python logging.

Example of usage:

::

  from torchelastic.tsm import events
  event = TsmEvent(..)
  events.record(event)

"""

import logging
import traceback
from types import TracebackType
from typing import Optional, Type

from torchelastic.tsm.events.handlers import get_logging_handler

from .api import SourceType, TsmEvent  # noqa F401

_events_logger = None


def _get_or_create_logger(destination: str = "null") -> logging.Logger:
    """
    Constructs python logger based on the destination type or extends if provided.
    Available destination could be found in ``handlers.py`` file.
    The constructed logger does not propagate messages to the upper level loggers,
    e.g. root logger. This makes sure that a single event can be processed once.

    Args:
        destination: The string representation of the event handler.
            Available handlers found in ``handlers`` module
        logger: Logger to be extended with the events handler. Method constructs
            a new logger if None provided.
    """
    global _events_logger
    if _events_logger:
        return _events_logger
    logging_handler = get_logging_handler(destination)
    _events_logger = logging.getLogger(f"tsm-events-{destination}")
    _events_logger.setLevel(logging.DEBUG)
    # Do not propagate message to the root logger
    _events_logger.propagate = False
    _events_logger.addHandler(logging_handler)
    return _events_logger


def record(event: TsmEvent, destination: str = "console") -> None:
    _get_or_create_logger(destination).info(event.serialize())


class log_event:
    """
    Context for logging tsm events. Creates TSMEvent and records it in
    the default destination at the end of the context execution. If exception occurs
    the event will be recorded as well with the error message.

    Example of usage:

    ::

    with log_event("api_name", ..):
        ...

    """

    def __init__(
        self,
        api: str,
        scheduler: Optional[str] = None,
        app_id: Optional[str] = None,
        runcfg: Optional[str] = None,
    ) -> None:
        self._tsm_event: TsmEvent = self._generate_tsm_event(
            api, scheduler or "", app_id, runcfg
        )

    def __enter__(self) -> "log_event":
        return self

    def __exit__(
        self,
        exec_type: Optional[Type[BaseException]],
        exec_value: Optional[BaseException],
        traceback_type: Optional[TracebackType],
    ) -> Optional[bool]:
        if traceback_type:
            self._tsm_event.raw_exception = traceback.format_exc()
        record(self._tsm_event)

    def _generate_tsm_event(
        self,
        api: str,
        scheduler: str,
        app_id: Optional[str] = None,
        runcfg: Optional[str] = None,
        source: SourceType = SourceType.UNKNOWN,
    ) -> TsmEvent:
        return TsmEvent(
            session=app_id or "",
            scheduler=scheduler,
            api=api,
            app_id=app_id,
            runcfg=runcfg,
            source=source,
        )
