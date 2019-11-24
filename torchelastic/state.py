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
    Represents the state of a trainer / tester. Each worker has an instance
    of the state object which is periodically synchronized to ensure that
    all workers have a consistent view of the world.

    The constructor of the state class should yield a state object
    representing the initial state, S_0. Hence, the constructor arguments
    should be those that are globally available to the workers without
    requiring a bootstrap communication.

    ``State`` is restorable, meaning that it can be captured at a point in time
    then restored to its original state. There are two sets of methods that enable
    this functionality: ``snapshot``/``apply`` and ``save``/``load``. Both
    ``snapshot`` and ``save`` captures enough data to restore the state
    from S_0. Typically only mutable fields of the state are captured in the
    snapshot and save.

    For example
    >>> class MyState(torchelastic.State):
    ...     def __init__(self, batch_size):
    ...         self.batch_size = batch_size
    ...         self.weights = [0] * 128
    ...
    >>> state = MyState(batch_size = 32)
    >>> do_work(state)

    the ``snapshot`` and ``save`` methods for ``MyState`` need only capture
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
        Figure out the latest state in the process group and
        broadcast it to all the workers.

        e.g., state could keep the which samples in the dataset have already been
              trained from on each worker. After state.sync(), it will know all the
              samples trained alreadys in the whole process group.

        """
        pass

    def snapshot(self):
        """
        Returns a user-defined object that can be used to restore this state
        to the point in time when this method was called. This method
        should be light weight when using the ``train_loop`` with ``rollback_enabled``
        since this method will be called before each ``train_step`` to be able
        to rollback from faults in the train_step.

        IMPORTANT: If the returned object is NOT ``torch.save`` compatible,
        then the ``save``  and ``load`` methods MUST be overridden
        for checkpointing to work correctly.

        Possible usage:
         >>> snapshot = state.snapshot()
         ... try:
         ...    do_something_that_modifies_state(state)
         ... except Exception:
         ...    # restore state (rollback + sync required)
         ...    state = state.rollback(snapshot)
         ...    state.sync()
        """

        return None

    def apply(self, snapshot) -> None:
        """
        Takes object returned by ``self.snapshot()`` and together with
        ``self.sync()``, restores the state to the point in time when
        the snapshot was taken. If rollback is enabled, then
        the train_loop calls this method followed by ``state.sync()``
        to first apply the snapshot to this object then re-initialize
        the runtime stack (e.g. data loaders)

        If the ``snapshot`` is ``None`` then it is interpreted as not
        supporting rollback and this method is a no-op.
        """

        if snapshot is not None:
            raise NotImplementedError(
                "Non-null snapshot provided. Must implement both snapshot and apply methods together"
            )

    def save(self, stream) -> None:
        """
        Writes the current state to the provided stream. If the ``snapshot``
        method does not yield a ``torch.save`` compatible object, then this
        method MUST be overridden to provide the correct save semantics
        for checkpointing to work correctly.

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

        snapshot = self.snapshot()
        torch.save(snapshot, stream)

    def load(self, stream) -> None:
        """
        Reads the captured state from the stream and applies it to this object.
        The `sync` method is expected to be called after load for full restoration.
        """

        snapshot = torch.load(stream)
        self.apply(snapshot)

    def should_save_checkpoint(self, rank):
        """
        NOTE: this method is subject to review and may be moved out of state.

        - Application need decide when to save checkpoint.
        eg: take checkpoint every x samples trained or every x seconds.

        - Every trainer need to return the same result.
        In DDP, backward pass call all_reduce to collect all gradient,
        this requires all trainers to anticipate. It might timeout if a trainer
        is stop and doing checkpoint while other worker complete the gradient
        computing. To prevent this, we put a barrier at the managed training loop
        while doing checkpoint, to make all trainers stop and waiting checkpoint
        complete.
        """
        return False
