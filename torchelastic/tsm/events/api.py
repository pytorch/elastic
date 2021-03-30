#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Union


class SourceType(str, Enum):
    UNKNOWN = "<unknown>"
    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"


@dataclass
class TsmEvent:
    """
    The class represents the event produced by tsm api calls.

    Arguments:
        session: Session id that was used to execute request.
        scheduler: Scheduler that is used to execute request
        api: Api name
        app_id: Unique id that is set by the underlying scheduler
        runcfg: Run config that was used to schedule app.
        source: Type of source the event is genereated.
    """

    session: str
    scheduler: str
    api: str
    app_id: Optional[str] = None
    runcfg: Optional[str] = None
    raw_exception: Optional[str] = None
    source: SourceType = SourceType.UNKNOWN

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "TsmEvent"]) -> "TsmEvent":
        if isinstance(data, TsmEvent):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
            if "source" in data_dict:
                # Convert string to enum
                try:
                    data_dict["source"] = SourceType(data_dict["source"])
                except ValueError:
                    data_dict.pop("source", None)

        return TsmEvent(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))
