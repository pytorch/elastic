#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torchelastic.tsm.driver.api as Scheduler
import torchelastic.tsm.driver.local_scheduler as local_scheduler


def get_scheduler(scheduler_type: str, **scheduler_params) -> Scheduler:
    schedulers = {"local": local_scheduler.create_scheduler}
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    scheduler_create_method = schedulers[scheduler_type]
    return scheduler_create_method(**scheduler_params)
