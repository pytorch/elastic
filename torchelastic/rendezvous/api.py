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
    Raised from ``RendezvousHandler.next_rendezvous()`` to signal that the
    rendezvous did not
    succeed within the allocated time. This is meant to be interpreted
    as a non-retryable type of failure.
    """

    pass


class RendezvousNonRetryableError(Exception):
    """
    Raised from any of the ``RendezvousHandler`` methods when a failure
    occured that should not be retried with the same worker process.
    """

    pass


class RendezvousHandler(abc.ABC):
    """
    Main rendezvous interface.

    .. note:: torchelastic users normally **do not** need to implement their
              own ``RendezvousHandler``. An implementation based on
              `etcd <https://etcd.io/>`__ is already provided, and is recommended
              for most users, provided they can deploy it in their environment.

    .. warning:: torchelastic is currently considered experimental,
                 so the APIs may change!
    """

    @abc.abstractmethod
    # pyre-fixme[11]: Annotation `Store` is not defined as a type.
    def next_rendezvous(self) -> Tuple["torch.distributed.Store", int, int]:
        """
        Main entry-point into the rendezvous barrier.
        Blocks until the rendezvous is complete (and the current
        process is included in the formed worker group), or a timeout occurs, or
        rendezvous was marked closed.

        Returns: a tuple of (``c10d Store``, ``rank``, ``world size``)

        Raises:
            RendezvousClosedException - if rendezvous for the current
               job is closed.
            RendezvousTimeoutException - on timeout
        """
        pass

    @abc.abstractmethod
    def is_closed(self) -> bool:
        """
        Checks whether rendezvous for current job has been closed,
        which means all future attempts to re-rendezvous (within same job) will
        fail.

        .. note:: ``is_closed`` and ``set_closed`` have semantics of eventual
                  propagation, and should not be used for synchronization.
                  The intention here is that if at least one worker decides
                  the job is finished, it will close the rendezvous, and
                  other workers will soon observe this and stop
                  training/rendezvous-ing as well.
        """
        pass

    @abc.abstractmethod
    def set_closed(self):
        """
        Used to mark the rendezvous (for current job) as closed.
        """
        pass

    @abc.abstractmethod
    def num_nodes_waiting(self) -> int:
        """
        Returns number of workers who *arrived late* at
        the rendezvous barrier, hence weren’t included in the current worker
        group.

        Callers should periodically call this method to check whether
        new members are waiting to join the job and if so admit them by
        calling ``next_rendezvous()`` (re-rendezvous).
        """
        pass
