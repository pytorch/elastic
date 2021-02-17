#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Optional, Union


EventMetadataValue = Union[str, int, float, bool, None]


class EventSource(str, Enum):
    """
    Known identifiers of the event producers.
    """

    AGENT = "AGENT"
    WORKER = "WORKER"


@dataclass
class Event:
    """
    The class represents the generic event that occurs during the torchelastic
    job execution. The event can be any kind of meaningful action.

    Args:
        name: event name.
        source: the event producer, e.g. agent or worker
        timestamp: timestamp in milliseconds when event occured.
        metadata: additional data that is associated with the event.
    """

    name: str
    source: EventSource
    timestamp: int = 0
    metadata: Dict[str, EventMetadataValue] = field(default_factory=dict)

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "Event"]) -> "Event":
        if isinstance(data, Event):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["source"] = EventSource[data_dict["source"]]
        return Event(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class TsmEvent:
    """
    The class represents the event produced by tsm api calls.

    Arguments:
        session: Session id that was used to execute request.
        scheduler: Scheduler that is used to execute request
        api: Api name
        unix_user: Unix user that executes request
        source_hostname: Hostname of server that executes request
        app_id: Unique id that is set by the underlying scheduler
        runcfg: Run config that was used to schedule app.
    """

    session: str
    scheduler: str
    api: str
    unix_user: str
    source_hostname: str
    app_id: Optional[str] = None
    runcfg: Optional[str] = None
    raw_exception: Optional[str] = None

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "TsmEvent"]) -> "TsmEvent":
        if isinstance(data, TsmEvent):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)

        return TsmEvent(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))
