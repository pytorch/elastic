#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torchelastic.checkpoint import (
    Checkpoint,
    CheckpointManager,
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

    def deep_copy(self):
        return copy.deepcopy(self)

    def should_save_checkpoint(self):
        return True

    def serialize(self, stream):
        # Dataset state we care about (i.e. need for resuming):
        dist_skip_index = self.dataset.dist_skip_index

        torch.save([dist_skip_index, self.total_sum], stream)

    def deserialize(self, stream):
        dist_skip_index, total_sum = torch.load(stream)

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        state = TestState()
        state.dataset.reinit(world_size, rank, start_index=dist_skip_index + 1)
        state.total_sum = total_sum
        state.set_worker_rank(rank)

        return state

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

    def deep_copy(self):
        # Need to increment `sync_counter` here otherwise the last increment
        # to `sync_counter` in `sync()` will be lost due to the rollback
        # in train_loop, and we'll just keep re-raising the excpetion.
        new_state = super().deep_copy()
        new_state.sync_counter += 1
        return new_state

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
    A special sub-class of TestState that disables rollback
    """

    def __init__(self):
        super().__init__()

    def deep_copy(self):
        # since deep_copy isn't supported, test will fail if it is called
        raise RuntimeError("deep_copy not supported")

    def rollback(self, state):
        raise RuntimeError("rollback not supported")

    def supports_rollback(self):
        return False


class TestWorkerStats(WorkerStats):
    """
    Simple test implementation of the WorkerStats interface
    """

    def __init__(self, progress_rate):
        self.progress_rate = progress_rate

    def get_progress_rate(self):
        return self.progress_rate


class TestCheckpoint(Checkpoint):
    def __init__(self, sequence_id, folder):
        self.checkpoint_folder = os.path.join(folder, str(sequence_id))
        # Create target Directory if it doesn't exist
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
            log.info("Created checkpoint folder: {}".format(self.checkpoint_folder))

    def open_output_stream(self, key):
        path = os.path.join(self.checkpoint_folder, key)
        log.info("saving checkpoint to: {}".format(path))
        return open(path, "wb")

    def open_input_stream(self, key):
        path = os.path.join(self.checkpoint_folder, key)
        log.info("Loading checkpoint from: {}".format(path))
        return open(path, "rb")

    def commit(self):
        pass

    def discard(self):
        pass


class TestCheckpointManager(CheckpointManager):
    """
    A CheckpointManager use current local folder
    """

    def __init__(self, folder):
        self.folder = folder

    def _get_sequence_ids(self):
        # r=root, d=directories, f = files
        def _is_int(input):
            try:
                int(input)
            except ValueError:
                return False
            return True

        folder_names = []
        for entry_name in os.listdir(self.folder):
            entry_path = os.path.join(self.folder, entry_name)
            if os.path.isdir(entry_path):
                folder_names.append(entry_name)

        seq_ids = [int(id) for id in folder_names if _is_int(id)]
        # sort desc
        seq_ids.sort(reverse=True)
        return seq_ids

    def create_checkpoint(self):
        """
        create a new checkpoint
        """
        seq_ids = self._get_sequence_ids()
        next_id = 0 if len(seq_ids) == 0 else seq_ids[0] + 1
        return TestCheckpoint(next_id, self.folder)

    def get_checkpoint(self, seqenceId):
        """
        get a specific checkpoint by sequence Id, return None if we cannot find
        it.
        """
        # type: (int) -> Checkpoint
        current_folder = os.path.join(self.folder, str(seqenceId))
        if not os.path.isdir(current_folder):
            raise Exception("folder: {} not found".format(current_folder))
        return TestCheckpoint(seqenceId, self.folder)

    def get_latest_checkpoint(self):
        """
        get the latest checkpoint. Return None if we don't have one
        """
        # type: () -> Checkpoint
        seq_ids = self._get_sequence_ids()
        if len(seq_ids) == 0:
            return None
        else:
            return self.get_checkpoint(seq_ids[0])

    def list_checkpoints(self):
        """
        list all available checkpoints
        LATTE might want this API.
        """
        # type: () -> list[Checkpoint]
        seq_ids = self._get_sequence_ids()
        return [self.get_checkpoint(id) for id in seq_ids]


@contextmanager
def test_checkpoint_manager(checkpoint_folder):
    # Code to acquire resource, e.g.:
    checkpoint_manager = TestCheckpointManager(checkpoint_folder)
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
