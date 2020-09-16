#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module


from torchelastic.multiprocessing.error_reporter.signal_handler import (
    LocalSignalHandler,
    SignalHandler,
)


def get_signal_handler() -> SignalHandler:
    return LocalSignalHandler()
