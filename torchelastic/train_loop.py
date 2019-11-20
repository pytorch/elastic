#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time

from torchelastic.checkpoint import CheckpointUtil
from torchelastic.coordinator import NonRetryableException, StopException
from torchelastic.metrics import get_elapsed_time_ms, publish_metric


log = logging.getLogger("TorchElasticTrainLoop")
log.setLevel(logging.INFO)

MAX_FAILURES = 100


def train(elastic_coordinator, train_step, state):
    """
        This is the main elastic data parallel loop. It starts from an initial 'state'.
        Each iteration calls 'train_step' and returns a new state. 'train_step'
        has the following interface:
            state, worker_stats = train_step(state)
        When 'train_step' exhausts all the data, a StopIteration exception should be
        thrown.
    """

    failure_count = 0
    rank = 0
    world_size = 1

    # TODO T55386785: load_checkpoint should be pulled out of CheckpointUtil
    checkpoint_util = CheckpointUtil(elastic_coordinator)

    # (TODO): Currently some clients, e.g., Classy Vision State (POC: aadock) is not
    #         a subclass of Elastic state class as not to interfers with their open
    #         source plan. In the future, when all the client states become subclasses,
    #         we should turn on this assersion check.
    # assert isinstance(state, State)

    supports_rollback = state.supports_rollback()
    log.info("rollback {0} supported".format("is" if supports_rollback else "isn't"))

    while not elastic_coordinator.should_stop_training():
        if failure_count >= MAX_FAILURES:
            # TODO(T43254350): print out the last error that occurred as well
            e = RuntimeError(
                "Exceeded max number of recoverable failures: {}".format(failure_count)
            )
            elastic_coordinator.on_error(e)
            raise e
        try:
            start_time = time.time()
            if supports_rollback:
                original_state = state.deep_copy()
            store, rank, world_size = elastic_coordinator.rendezvous_barrier()
            elastic_coordinator.init_process_group()

            # load checkpoint if necessary
            # TODO: Refactor: checkpoint_util.set_checkpoint_loaded() after
            # checkpoint_util.load_checkpoint(state, rank) seems not elegant
            state = checkpoint_util.load_checkpoint(state, rank)

            state_sync_start_time = time.time()
            state.sync(world_size, rank)
            publish_metric(
                "torchelastic",
                "state_sync.duration.ms",
                get_elapsed_time_ms(state_sync_start_time),
            )
            checkpoint_util.set_checkpoint_loaded()
            log.info("Rank {0} synced state with other nodes".format(rank))
        except StopException:
            log.info("Rank {0} received stopped signal. Exiting training.".format(rank))
            break
        except RuntimeError as e:
            # TODO(T43254350): We may want to be more discriminating than
            # `RuntimeError` here
            elastic_coordinator.on_error(e)
            if supports_rollback:
                state = state.rollback(original_state)
            failure_count += 1
            continue
        except (NonRetryableException, Exception) as e:
            elastic_coordinator.on_error(e)
            raise
        finally:
            publish_metric(
                "torch_elastic",
                "outer_train_loop.duration.ms",
                get_elapsed_time_ms(start_time),
            )

        # Note that the loop might not even start if the rendezvous was closed
        # due to one of the trainer processes completing earlier.
        while not elastic_coordinator.should_stop_training():
            try:
                start_time = time.time()
                if supports_rollback:
                    original_state = state.deep_copy()

                train_step_start_time = time.time()
                state, worker_stats = train_step(state)
                publish_metric(
                    "torchelastic",
                    "train_step.duration.ms",
                    get_elapsed_time_ms(train_step_start_time),
                )

                elastic_coordinator.monitor_progress(state, worker_stats)

                checkpoint_util.save_checkpoint(state, rank)
                if elastic_coordinator.should_rendezvous(state):
                    log.info("Rank {0} will re-rendezvous".format(rank))
                    # Executor told us, for whatever reason, to re-rendezvous.
                    # This can occur if another node encounters an error,
                    # if a new node becomes available to train,
                    # or potentially even if it's time to checkpoint.
                    break
                elastic_coordinator.report_progress(state)
            except StopIteration:
                log.info("Rank {0} finished all the iterations".format(rank))
                # Current trainer process completed processing assigned subset of
                # examples. Other trainer processes need to stop as well.
                # This sends an explicit signal on training completion.
                elastic_coordinator.signal_training_done()
                break
            except RuntimeError as e:
                # TODO(T43254350): We may want to be more discriminating than
                # `RuntimeError` here
                elastic_coordinator.on_error(e)
                if supports_rollback:
                    state = state.rollback(original_state)
                failure_count += 1
                break
            except Exception as e:
                elastic_coordinator.on_error(e)
                raise
            finally:
                publish_metric(
                    "torchelastic",
                    "inner_train_loop.duration.ms",
                    get_elapsed_time_ms(start_time),
                )

    if elastic_coordinator.should_stop_training():
        return state
    else:
        # This is an error condition and should not happen.
        raise Exception(
            "Exiting without training complete. rank: {0},"
            " should_stop_training: {1}".format(
                rank, elastic_coordinator.should_stop_training()
            )
        )
