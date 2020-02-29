#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
from typing import List

import torchelastic.distributed as edist
import torchelastic.metrics as metrics


log = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_KEY = "default"

_CHECKPOINT_MANAGER = None


def set_checkpoint_manager(checkpoint_manager):
    global _CHECKPOINT_MANAGER
    _CHECKPOINT_MANAGER = checkpoint_manager


def get_checkpoint_manager():
    global _CHECKPOINT_MANAGER
    return _CHECKPOINT_MANAGER


class _CheckpointBarrier(object):
    """
    Checkpoint Barrier
    """

    def __init__(self, rank, coordinator):
        self.rank = rank
        self.coordinator = coordinator

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # We put a explicit barrier here to make sure all trainer sync
        # after checkpoint was loaded
        log.info(f"Rank {self.rank} enter checkpoint barrier")
        self.coordinator.barrier()
        log.info(f"Rank {self.rank} exit checkpoint barrier")


class CheckpointUtil:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.checkpoint_manager = get_checkpoint_manager()
        self.checkpoint_loaded = False

    def _do_load_checkpoint(self, state):
        """
        Loads the checkpoint.
        Construct a new state object if checkpoint existed.
        """

        try:
            log.info("Finding latest available checkpoint...")
            checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if not checkpoint:
                log.info("Cannot find a checkpoint to load.")
                return state
            else:
                log.info("Loading checkpoint...")
                with checkpoint.open_input_stream(_DEFAULT_CHECKPOINT_KEY) as stream:
                    # Start from simple with _DEFAULT_CHECKPOINT_KEY
                    state.load(stream)
                    log.info("Load checkpoint successfully.")
                    return state
        except Exception as e:
            log.error("Load checkpoint fail: {}".format(e))
            raise e

    @metrics.profile("torchelastic")
    def load_checkpoint(self, state, rank: int):
        """
        Loads checkpoint if the checkpoint manager has been configured and
        at least one worker has already loaded the checkpoint
        """
        if not self.checkpoint_manager:
            # checkpoint not enabled
            return state

        # all gather `checkpoint_loaded` from all trainers, return true
        # if any trainer have ever loaded checkpoint
        any_checkpoint_loaded = (
            edist.all_gather_return_max_long(1 if self.checkpoint_loaded else 0) == 1
        )

        if any_checkpoint_loaded:
            # checkpoint already loaded by one of the existing trainer
            return state

        # we load checkpoint only if all trainers start from scratch. it is
        # not necessary to load checkpoint if there is a good trainer as new
        # trainer can sync state from it.
        # Start with simple scenario, we always ask one single trainer to
        # load checkpoint and other trainer sync from it
        if rank == 0:
            state = self._do_load_checkpoint(state)

        return state

    def set_checkpoint_loaded(self):
        """
        Indicate checkpoint have been loaded
        """
        self.checkpoint_loaded = True

    def _do_save_checkpoint(self, state):
        """
        Save checkpoint.
        """
        checkpoint = None
        try:
            log.info("Creating new checkpoint...")
            checkpoint = self.checkpoint_manager.create_checkpoint()
            log.info("Saving checkpoint...")
            with checkpoint.open_output_stream(_DEFAULT_CHECKPOINT_KEY) as stream:
                # Start from simple with _DEFAULT_CHECKPOINT_KEY
                state.save(stream)
                log.info("Save Checkpoint successfully.")
                checkpoint.commit()
        except Exception as e:
            log.error("Save checkpoint fail: {}".format(e))
            if not checkpoint:
                # discard bad checkpoint
                checkpoint.discard()
            raise e

    @metrics.profile("torchelastic")
    def save_checkpoint(self, state, rank: int):
        """
        TODO: https://github.com/pytorch/elastic/issues/9
        """
        if (
            self.checkpoint_manager  # checkpoint enabled
            and (
                self.coordinator.should_save_checkpoint()
                or state.should_save_checkpoint(rank)
            )
            # ASSUMPTION: `state.should_save_checkpoint()` return
            # consistent value for all workers.
        ):
            # we will save checkpoint if coordinator/platform told us
            # or the application told us to do.
            # ASSUMPTION: PET built on DDP, which has an implicit barrier
            # (reduce_all) in train_step(state). State are all good when
            # We come here, otherwise it will break out of the loop if any
            # exception is raised.
            with _CheckpointBarrier(rank, self.coordinator):
                if rank == 0:
                    self._do_save_checkpoint(state)


class Checkpoint(abc.ABC):
    """
    Represent a checkpoint record. Checkpoint object encapsulate all storage details.
    User don't need to know where to save the data and how to save the data.
    It provide a key/value storage and support both synchronize/streaming way
    to storage data.
    """

    @abc.abstractmethod
    def open_output_stream(self, key):
        """
        with a key/value interface, user have the flexibility to save their state
        in one single key or sharding the state in parallel to get better
        performance or even save different part of the state in different worker
        in a distributed fashion. Re-sharding is also available as we can always
        over partition the key and re-regroup it later.

        Note:
            - Same key will be override if call the method twice.
            - The method could fail, client need to handle the exception
        """
        pass

    @abc.abstractmethod
    def open_input_stream(self, key):
        pass

    @abc.abstractmethod
    def commit(self):
        """
        the checkpoint only available after commit is called. The related checkpoint
        will be saved after this.

        This method is idempotent will ignore the call if its already commit.
        """
        pass

    @abc.abstractmethod
    def discard(self):
        """
        Discard current checkpoint.
        TODO: https://github.com/pytorch/elastic/issues/9
        """
        pass


class CheckpointManager(abc.ABC):
    """
    Some use-cases need to access the latest "k" checkpoints
    the checkpoint manager give a way to manage and access all the checkpoints
    """

    @abc.abstractmethod
    def create_checkpoint(self) -> Checkpoint:
        """
        create a new checkpoint
        """
        pass

    @abc.abstractmethod
    def get_checkpoint(self, sequence_id: int) -> Checkpoint:
        """
        get a specific checkpoint by sequence Id, return None if we cannot find
        it.
        """
        pass

    @abc.abstractmethod
    def get_latest_checkpoint(self) -> Checkpoint:
        """
        get the latest checkpoint. Return None if we don't have one
        """
        pass

    @abc.abstractmethod
    def list_checkpoints(self) -> List[Checkpoint]:
        """
        list all available checkpoints.
        """
        pass
