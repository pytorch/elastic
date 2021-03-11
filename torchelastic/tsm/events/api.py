#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass
from typing import Optional, Union


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
    """

    session: str
    scheduler: str
    api: str
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
