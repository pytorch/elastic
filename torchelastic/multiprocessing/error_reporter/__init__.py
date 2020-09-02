#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module


from .api import configure, exec_fn, get_error, get_platform  # noqa F401


try:
    # @manual
    from torchelastic.multiprocessing.error_reporter.static_init import *  # noqa: F401 F403
except ModuleNotFoundError:
    pass
