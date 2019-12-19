#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torchelastic.checkpoint import (
    FileSystemCheckpointManager,
    get_checkpoint_manager,
    set_checkpoint_manager,
)
from torchelastic.p2p.coordinator_p2p import CoordinatorP2P
from torchelastic.state import State
from torchelastic.worker_stats import WorkerStats


log = logging.getLogger(__name__)


class TestDataset:
    def __init__(self):
        self.data = range(11, 31)
        self.start_index = 0

        # All data up to "skip index" is considered previously read. This is
        # used to figure out where we need to "resume" from in case of reinit.
        self.local_skip_index = -1
        self.dist_skip_index = -1  # Max across all local_skip_index values.

    def reinit(self, world_size, rank, start_index):
        self.world_size = world_size
        self.rank = rank
        self.start_index = start_index

        self.local_skip_index = start_index - 1
        self.dist_skip_index = start_index - 1

        self.indices = []
        for idx in range(self.start_index + rank, len(self.data), self.world_size):
            self.indices.append(idx)

        self.iter = iter(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        index = next(self.iter)
        self.local_skip_index = index
        self.dist_skip_index = self._distributed_next_index()
        return self.data[index]

    def _distributed_next_index(self):
        tensor = torch.tensor([self.local_skip_index])
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor[0])


class TestState(State):
    def __init__(self):
        self.total_sum = 0
        self.nums = []
        self.dataset = TestDataset()
        self._worker_rank = None

    def set_worker_rank(self, rank):
        """
        sets the rank of the worker that returned this state from the train_step
        used in tests for asserting behavior/results from a specific worker
        this field does and should not be serialized/deserialized
        """
        self._worker_rank = rank

    def get_worker_rank(self):
        return self._worker_rank

    def should_save_checkpoint(self, rank):
        return True

    def capture_snapshot(self):
        snapshot = {}
        snapshot["total_sum"] = self.total_sum
        snapshot["dataset.dist_skip_index"] = self.dataset.dist_skip_index
        return snapshot

    def apply_snapshot(self, snapshot):
        self.total_sum = snapshot["total_sum"]
        dist_skip_index = snapshot["dataset.dist_skip_index"]
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        self.dataset.reinit(world_size, rank, start_index=dist_skip_index + 1)

    def get_data_iterator(self):
        return iter(self.dataset)

    def sync(self, world_size, rank):
        src_rank = 0
        tensor = torch.LongTensor([self.dataset.dist_skip_index, self.total_sum])
        dist.broadcast(tensor, src=src_rank)

        self.dataset.reinit(world_size, rank, start_index=int(tensor[0]) + 1)
        self.total_sum = int(tensor[1])
        self.set_worker_rank(dist.get_rank())


class TestStateFailOnSync(TestState):
    """
    A special sub-class of TestState that throws a specific exception
    on the 3rd call to sync().
    """

    def __init__(self, exception_type, exception_msg):
        self.sync_counter = 0
        self.exception_type = exception_type
        self.exception_msg = exception_msg
        super().__init__()

    def capture_snapshot(self):
        # Need to increment `sync_counter` here otherwise the last increment
        # to `sync_counter` in `sync()` will be lost due to the rollback
        # in train_loop, and we'll just keep re-raising the exception.
        snapshot = super().capture_snapshot()
        snapshot["sync_counter"] = self.sync_counter + 1
        return snapshot

    def apply_snapshot(self, snapshot):
        super().apply_snapshot(snapshot)
        self.sync_counter = snapshot["sync_counter"]

    def sync(self, world_size, rank):
        self.sync_counter += 1
        # The magic number 3 is here because iteration 1 is the initial
        # sync, iteration 2 occurs in the first call to `_train_step`,
        # and we want to fail on iteration 3 (first re-rendezvous).
        if self.sync_counter == 3:
            raise self.exception_type(self.exception_msg)

        super().sync(world_size, rank)


class TestStateWithRollbackDisabled(TestState):
    """
    A special sub-class of TestState that disables rollback.
    Rollback is implicitly disabled if snapshot returns None
    and apply is a no-op.

    """

    def __init__(self):
        super().__init__()

    def capture_snapshot(self):
        return None

    def apply_snapshot(self, snapshot):
        pass


class TestWorkerStats(WorkerStats):
    """
    Simple test implementation of the WorkerStats interface
    """

    def __init__(self, progress_rate):
        self.progress_rate = progress_rate

    def get_progress_rate(self):
        return self.progress_rate


@contextmanager
def test_checkpoint_manager(checkpoint_folder):
    # Code to acquire resource, e.g.:
    checkpoint_manager = FileSystemCheckpointManager(checkpoint_folder)
    old_checkpoint_manager = get_checkpoint_manager()
    set_checkpoint_manager(checkpoint_manager)
    yield
    set_checkpoint_manager(old_checkpoint_manager)


class TestCoordinatorP2P(CoordinatorP2P):
    def __init__(
        self, c10d_backend, init_method, max_num_trainers, process_group_timeout
    ):
        super(TestCoordinatorP2P, self).__init__(
            c10d_backend, init_method, max_num_trainers, process_group_timeout
        )

    def should_save_checkpoint(self):
        """
        This normally happens when the job was explicitly ask for checkpoint.
        eg: executor got a preemption from scheduler
        """
        return True
