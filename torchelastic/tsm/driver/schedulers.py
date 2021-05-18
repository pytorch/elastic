#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import torchx.schedulers.local_scheduler as local_scheduler
from torchx.specs.api import Scheduler, SchedulerBackend


def get_schedulers(
    session_name: str, **scheduler_params
) -> Dict[SchedulerBackend, Scheduler]:
    return {
        "local": local_scheduler.create_scheduler(session_name, **scheduler_params),
        "default": local_scheduler.create_scheduler(session_name, **scheduler_params),
    }
