#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc


class StopException(Exception):
    """
    Raised to signal that the Coordinator wants the caller to exit cleanly.
    """

    pass


class NonRetryableException(Exception):
    """
    Raised to signal that the Coordinator issued an unclean shutdown
    and the callers should not retry on it.
    """

    pass


class Coordinator(abc.ABC):
    """
    Abstract base elastic coordinator class to insulate implementation of
    coordination code from workflow code.
    """

    @abc.abstractmethod
    def __init__(self):
        """
        Derived versions of this class should configure rendezvous parameters
        in the constructor, e.g. min_nodes, max_nodes, rendezvous timeout,
        rendezvous timeout for min nodes, collective timeout etc
        """

        pass

    @abc.abstractmethod
    def rendezvous_barrier(self):
        """
        Acquire the next version of rendezvous.
        Return Tuple[store(c10dStore), rank(int), world_size(int)]
        """
        pass

    @abc.abstractmethod
    def barrier(self):
        """
        A regular barrier (no rendezvous) for synchronizing trainers.
         - This method throws a RuntimeError if not all workers reach join the
           barrier within a specified timeout. The timeout value is implementation
           specific.
         - If `barrier()` throws in one trainer, other trainers should throw.
        """
        pass

    @abc.abstractmethod
    def init_process_group(self):
        """
        Creates a ProcessGroup which manages collective and p2p communication
        for a distributed application.

        Returns ProcessGroup
        """

        pass

    @abc.abstractmethod
    def should_save_checkpoint(self):
        """
        Whether the PET training loop need to do checkpoint.
        This normally happens when the job was explicitly ask for checkpoint.
        eg: coordinator got a preemption from scheduler
        """
        pass

    @abc.abstractmethod
    def should_rendezvous(self, state):
        """
        Queries the coordinator to see if the job should stop working and
        yield control back to the coordinator to re-rendezvous. This often occurs
        on membership change, checkpoint, etc.
        """

        pass

    @abc.abstractmethod
    def should_stop_training(self):
        """
        Queries the coordinator to see if the trainer should break out of the
        PET control loop either because the training has been completed or a
        global termination was requested.
        """

        pass

    @abc.abstractmethod
    def signal_training_done(self):
        """
        Signals the coordinator that the training job has been completed.
        This also closes open rendezvous and destroys current process group.
        """

        pass

    @abc.abstractmethod
    def monitor_progress(self, state, worker_stats):
        """
        Inspect worker progress and perform corrective actions, if needed.
        """

        pass

    @abc.abstractmethod
    def report_progress(self, state):
        """
        Report forward job progress.
        Can also be used as a watchdog to detect stuck/hung/missing nodes.
        """

        pass

    @abc.abstractmethod
    def on_error(self, e):
        """
        Report an error / invoke error handling logic.
        """

        pass
