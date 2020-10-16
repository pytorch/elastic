#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

# TODO(aivanou): currently the api is exposed in a single place and
# it is hard to distinguish what needs to be executed on parent process and
# what needs to be executed on child process.
from .api import (  # noqa F401
    ErrorMessage,
    ErrorType,
    cleanup,
    exec_fn,
    get_error,
    record,
)
