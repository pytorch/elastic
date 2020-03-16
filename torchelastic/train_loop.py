#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import warnings

import torchelastic
from torchelastic.checkpoint import CheckpointUtil
from torchelastic.coordinator import NonRetryableException, StopException
from torchelastic.metrics import get_elapsed_time_ms, publish_metric


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MAX_FAILURES = 100


def to_generator(train_step_fn):
    """
        Makes the provided train step function a Python generator.
        The stop condition for the returned generator is when the
        train_step_fn raises a ``StopIteration``, which is interpreted
        as end of data.
    """

    def wrap(state):
        while True:
            try:
                yield train_step_fn(state)
            except StopIteration as e:
                logging.info("End of data reached from train_step", exc_info=e)
                return

    return wrap


def train(elastic_coordinator, train_step, state):
    """
        This function defines a train_loop as well as a python generate by iterate '
        train_step' function, the 'train_step' has the following interface:
            state, worker_stats = train_step(state)
        When 'train_step' exhausts all the data, a StopIteration exception should be
        thrown.
    """
    warnings.warn("Deprecated, use run_train() instead", DeprecationWarning)
    return run_train(elastic_coordinator, to_generator(train_step), state)


def run_train(coordinator, train_step_gen, state):
    """
        Elastic data parallel loop. Iteratively calls ``train_step_gen`` on the
        provided ``state`` object. The generator is allowed to return
        ``(state, worker_stats)``.
        Note: since the returned ``state`` from the generator will be ignored
        and is kept for backwards compatibility reasons.
        See: https://github.com/pytorch/elastic/issues/48
    """

    assert isinstance(state, torchelastic.State)
    assert isinstance(coordinator, torchelastic.Coordinator)

    failure_count = 0
    rank = 0

    checkpoint_util = CheckpointUtil(coordinator)

    while not coordinator.should_stop_training():
        # See: https://github.com/pytorch/elastic/issues/7
        if failure_count >= MAX_FAILURES:
            e = RuntimeError(
                "Exceeded max number of recoverable failures: {}".format(failure_count)
            )
            coordinator.on_error(e)
            raise e

        start_time = time.time()
        snapshot = state.capture_snapshot()

        try:
            store, rank, world_size = coordinator.rendezvous_barrier()
            coordinator.init_process_group()
            # Sync befor continue any user code since init_process_group
            # does not sync.
            coordinator.barrier()

            # load checkpoint if necessary
            state = checkpoint_util.load_checkpoint(state, rank)

            state_sync_start_time = time.time()
            state.sync(world_size, rank)
            publish_metric(
                "torchelastic",
                "state_sync.duration.ms",
                get_elapsed_time_ms(state_sync_start_time),
            )
            checkpoint_util.set_checkpoint_loaded()
            coordinator.barrier()
            log.info("Rank {0} synced state with other nodes".format(rank))
        except StopException:
            log.info("Rank {0} received stopped signal. Exiting training.".format(rank))
            break
        except RuntimeError as e:
            # See: https://github.com/pytorch/elastic/issues/7
            coordinator.on_error(e)
            state.apply_snapshot(snapshot)
            failure_count += 1
            continue
        except (NonRetryableException, Exception) as e:
            coordinator.on_error(e)
            raise
        finally:
            publish_metric(
                "torchelastic",
                "outer_train_loop.duration.ms",
                get_elapsed_time_ms(start_time),
            )

        # Note that the loop might not even start if the rendezvous was closed
        # due to one of the trainer processes completing earlier.
        generator = train_step_gen(state)
        while not coordinator.should_stop_training():
            start_time = time.time()
            snapshot = state.capture_snapshot()

            try:
                train_step_start_time = time.time()
                _, worker_stats = next(generator)
                publish_metric(
                    "torchelastic",
                    "train_step.duration.ms",
                    get_elapsed_time_ms(train_step_start_time),
                )

                coordinator.monitor_progress(state, worker_stats)

                checkpoint_util.save_checkpoint(state, rank)
                if coordinator.should_rendezvous(state):
                    log.info("Rank {0} will re-rendezvous".format(rank))
                    # Executor told us, for whatever reason, to re-rendezvous.
                    # This can occur if another node encounters an error,
                    # if a new node becomes available to train,
                    # or potentially even if it's time to checkpoint.
                    break
                coordinator.report_progress(state)
            except StopIteration:
                log.info("Rank {0} finished all the iterations".format(rank))
                # Current trainer process completed processing assigned subset of
                # examples. Other trainer processes need to stop as well.
                # This sends an explicit signal on training completion.
                coordinator.signal_training_done()
                break
            except RuntimeError as e:
                # See: https://github.com/pytorch/elastic/issues/7
                coordinator.on_error(e)
                state.apply_snapshot(snapshot)
                failure_count += 1
                break
            except Exception as e:
                coordinator.on_error(e)
                raise
            finally:
                publish_metric(
                    "torchelastic",
                    "inner_train_loop.duration.ms",
                    get_elapsed_time_ms(start_time),
                )

    if coordinator.should_stop_training():
        return state
    else:
        # This is an error condition and should not happen.
        raise Exception(
            "Exiting without training complete. rank: {0},"
            " should_stop_training: {1}".format(
                rank, coordinator.should_stop_training()
            )
        )
