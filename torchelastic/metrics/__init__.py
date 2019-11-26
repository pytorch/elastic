#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import (  # noqa F401
    ConsoleMetricHandler,
    MetricHandler,
    NullMetricHandler,
    configure,
    get_elapsed_time_ms,
    getStream,
    profile,
    publish_metric,
)


def initialize_metrics():
    pass


try:
    from torchelastic.metrics.static_init import *  # noqa: F401 F403
except ModuleNotFoundError:
    pass
