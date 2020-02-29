#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch


class State(abc.ABC):
    """
    Represents the state of a trainer. Each worker has an instance
    of the state object which is periodically synchronized to ensure that
    all workers have a consistent view of the world.

    The constructor of the state class should yield a state object
    representing the initial state, S_0. Hence, the constructor arguments
    should be those that are globally available to the workers without
    requiring a bootstrap communication.

    ``State`` is restorable, meaning that it can be captured at a point in time
    then restored to its original state. There are two sets of methods that enable
    this functionality: ``capture_snapshot``/``apply`` and ``save``/``load``. Both
    ``capture_snapshot`` and ``save`` captures enough data to restore the state
    from S_0. Typically only mutable fields of the state are captured in the
    capture_snapshot and save.

    For example
    >>> class MyState(torchelastic.State):
    ...     def __init__(self, batch_size):
    ...         self.batch_size = batch_size
    ...         self.weights = [0] * 128
    ...
    >>> state = MyState(batch_size = 32)
    >>> do_work(state)

    the ``capture_snapshot`` and ``save`` methods for ``MyState`` need only capture
    ``weights`` since the ``batch_size`` is passed as a constructor argument
    and hence it is assumed to be globally available. Typically the constructor
    arguments are direct or derived parameters to the training job.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def sync(self, world_size, rank):
        """
        This method is responsible for synchronizing the state object
        amongst all workers. It is also responsible for ensuring that the
        state object is rendered "usable", for instance, if member variables of
        this object requires (re)-initialization it should be done
        in this method.

        This method is expected to be called from all the workers collectively
        (e.g. not just called on a subset of workers, or called at widely different
        times). Torchelastic calls this method on events that may render states
        of one or more workers "out-of-sync" or "corrupt" when compared to the
        rest of the workers. Examples of these events are:
          * on worker initialization
          * on membership changes
          * on rollback

        """
        pass

    def capture_snapshot(self):
        """
        Returns a user-defined object that can be used to restore this state
        to the point in time when this method was called. This method
        should be light weight when using the ``train_loop`` with
        ``rollback_enabled`` since this method will be called before each
        ``train_step`` to be able to rollback from faults in the train_step.

        IMPORTANT: If the returned object is NOT ``torch.save`` compatible,
        then the ``save``  and ``load`` methods MUST be overridden
        for checkpointing to work correctly.

        Possible usage:
         >>> snapshot = state.capture_snapshot()
         ... try:
         ...    do_something_that_modifies_state(state)
         ... except Exception:
         ...    # restore state (rollback + sync required)
         ...    state = state.apply_snapshot(snapshot)
         ...    state.sync()
        """

        return None

    def apply_snapshot(self, capture_snapshot) -> None:
        """
        Takes object returned by ``self.capture_snapshot()``,
        restores the state to its value when the snapshot was taken. Note that
        torchelastic always calls ``state.sync()`` after ``state.apply_snapshot()``
        to ensure that any re-initialization of member variables can take place
        and the state object is synchronized properly among all workers.

        This method should be a no-op if the ``capture_snapshot`` is
        trivial (e.g. None). torchelastic reserves the right to bypass
        the calling of this method if the ``capture_snapshot()``
         method returns ``None``.
        """

        if capture_snapshot is not None:
            raise NotImplementedError(
                "Non-null capture_snapshot provided."
                " Must implement both"
                " capture_snapshot and apply_snapshot methods together"
            )

    def save(self, stream) -> None:
        """
        Writes the current state to the provided stream. If the ``capture_snapshot``
        method does not yield a ``torch.save`` compatible object, then this
        method MUST be overridden to provide the correct save semantics
        for checkpointing to work correctly.

        The ``stream`` parameter is a ``torch.save`` compatible file-like
        object.

        Possible checkpoint implementation:
         >>> def save_checkpoint(state, filename):
         ...    with open(filename, "w") as f:
         ...        state.save(f)
         ...
         >>> def load_checkpoint(state, filename):
         ...    with open(filename, "r") as f:
         ...        state.load(f)
         ...
        """

        snapshot = self.capture_snapshot()
        torch.save(snapshot, stream)

    def load(self, stream) -> None:
        """
        Reads the captured state from the stream and applies it to this object.
        The ``sync()`` method is expected to be called after load for full restoration.

        The ``stream`` parameter is a ``torch.load`` compatible file-like
        object.
        """

        capture_snapshot = torch.load(stream)
        self.apply_snapshot(capture_snapshot)

    def should_save_checkpoint(self, rank):
        """
        NOTE: this method is subject to review and may be moved out of state.

        Returns a boolean value indicating whether a checkpoint should
        be saved. Torchelastic calls this method at the end of each
        ``train_step``. This method can be implemented to checkpoint
        on `k` train steps (e.g. on each epoch).

        IMPORTANT: all workers should return the same value for this method
        since torchelastic has an implicit barrier when performing checkpoints
        hence even if the checkpoint is being saved from 1/n workers all workers
        need to enter the checkpoint barrier to ensure that they begin the
        next iteration of train_step together. Otherwise, the collectives
        operation may fail due to a timeout since 1/n workers will fall behind
        due to the checkpoint operation while others have moved on to the next
        round of ``train_step``.
        """

        return False
