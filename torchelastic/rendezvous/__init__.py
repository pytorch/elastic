#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import (  # noqa: F401
    RendezvousClosedException,
    RendezvousHandler,
    RendezvousNonRetryableError,
    RendezvousTimeoutException,
)


try:
    import torchelastic.rendezvous.static_init  # noqa: F401
except ModuleNotFoundError:
    pass
