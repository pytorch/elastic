#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchelastic.tsm.driver.api import (  # noqa: F401 F403
    Application,
    AppState,
    Container,
    Resources,
    Role,
    RunMode,
    Session,
)
from torchelastic.tsm.driver.schedulers import get_scheduler
from torchelastic.tsm.driver.standalone_session import (  # noqa: F401 F403
    StandaloneSession,
)


def session(
    name: str, scheduler_type: str, backend: str = "standalone", **scheduler_args
):
    scheduler = get_scheduler(scheduler_type, **scheduler_args)
    if backend != "standalone":
        raise ValueError(
            f"Unsupported session backend: {backend}. Supported values: standalone"
        )
    return StandaloneSession(name=name, scheduler=scheduler)
