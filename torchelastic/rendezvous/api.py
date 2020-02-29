#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Tuple


class RendezvousClosedException(Exception):
    """
    Raised when a rendezvous for the specified run_id is closed.
    This is used to signal completion to nodes that arrive late.
    """

    pass


class RendezvousTimeoutException(Exception):
    """
    Raised from `next_rendezvous` to signal that the rendezvous did not
    succeed within the allocated time. This is meant to be interpreted
    as a non-retryable type of failure.
    """

    pass


class RendezvousNonRetryableError(Exception):
    """
    Raised from any of the `RendezvousHandler` methods when a failure
    occured that should not be retried with the same worker process.
    """

    pass


class RendezvousHandler(abc.ABC):
    @abc.abstractmethod
    def next_rendezvous(self) -> Tuple["torch.distributed.Store", int, int]:
        """
        Returns a tuple of (c10d Store, rank, world size),
        or raises RendezvousClosedException,
        or raises RendezvousTimeoutException.
        """
        pass

    @abc.abstractmethod
    def is_closed(self) -> bool:
        pass

    @abc.abstractmethod
    def set_closed(self):
        pass

    @abc.abstractmethod
    def num_nodes_waiting(self) -> int:
        pass
