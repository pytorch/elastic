#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import socket
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

# Register handler for etcd-based rendezous:
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
from torchelastic import metrics
from torchelastic.coordinator import Coordinator, NonRetryableException, StopException
from torchelastic.event_logger import get_event_logger
from torchelastic.rendezvous import (
    RendezvousClosedException,
    RendezvousHandler,
    RendezvousTimeoutException,
)


# Logger
log = logging.getLogger("CoordinatorP2P")
log.setLevel(logging.INFO)


class CoordinatorP2P(Coordinator):
    # monitoring progress is expensive as it requires a collective communication
    # operation. MONITOR_PROGRESS_FREQ controls how often we run this.
    MONITOR_PROGRESS_FREQ = 1000

    def __init__(
        self,
        c10d_backend,
        init_method,
        max_num_trainers,
        process_group_timeout=10000,
        coordinator_pg_timeout=600000,  # default 10 mins for coordinator pg timeout
    ):
        self.c10d_backend = c10d_backend
        self.init_method = init_method
        self.rendezvous = dist.rendezvous(init_method)
        assert isinstance(
            self.rendezvous, RendezvousHandler
        ), "CoordinatorP2P requires a torchelastic.rendezvous.RendezvousHandler"
        assert coordinator_pg_timeout > process_group_timeout, (
            "coordinator_pg_timeout {} (ms) must larger than or equal to "
            "process_group_timeout {} (ms)".format(
                coordinator_pg_timeout, process_group_timeout
            )
        )
        self.max_num_trainers = max_num_trainers
        self.process_group_timeout = process_group_timeout
        self.coordinator_pg_timeout = coordinator_pg_timeout
        self.rank = -1
        self.world_size = 0
        self.is_worker_straggler = False
        self.stop_training = False
        self.coordinator_process_group = None
        self.monitor_progress_step = 0
        self.host_name = socket.gethostname()
        self.pid = os.getpid()
        self.event_logger = get_event_logger()
        metrics.initialize_metrics()

    def _log_event(self, event_name, message=None):
        if message is None:
            message = {}
        message["event_name"] = event_name
        message["host_name"] = self.host_name
        message["pid"] = self.pid
        message["rank"] = self.rank
        self.event_logger.log_event(event_name, json.dumps(message))

    def _destroy_process_group(self):
        if dist.is_initialized():
            if self.c10d_backend != dist.Backend.GLOO:
                dist.destroy_process_group(self.coordinator_process_group)
            dist.destroy_process_group()

    @metrics.profile("torchelastic")
    def rendezvous_barrier(self):
        self._destroy_process_group()
        try:
            self._log_event("rendezvous_started")
            self.store, self.rank, self.world_size = self.rendezvous.next_rendezvous()
            self._log_event("rendezvous_succeeded", {"word_size": self.world_size})
        except RendezvousClosedException:
            # Sets the local variable to True
            self._log_event("rendezvous_closed")
            self.stop_training = True
            raise StopException(
                "Rank {0} received RendezvousClosedException."
                " Raising a StopException".format(self.rank)
            )
        except RendezvousTimeoutException as e:
            self._log_event("rendezvous_failed_timeout")
            raise NonRetryableException(
                "Rank {0} received a timeout Exception. "
                "This indicates that workers were permanently stuck."
                "Make sure that you have available resources. "
                "Detailed message: {1}".format(self.rank, str(e))
            )
        except Exception as e:
            self._log_event("rendezvous_failed")
            raise NonRetryableException(
                "Rank {0} received an Exception."
                " Detailed message: {1}".format(self.rank, str(e))
            )
        log.info(
            "Got next rendezvous: rank {0}, world size {1}".format(
                self.rank, self.world_size
            )
        )

        # Assume straggler state is unreliable after rendezvous
        self.is_worker_straggler = False

        return self.store, self.rank, self.world_size

    def barrier(self):
        # Use gloo process group to implement a barrier in case NCCL get stuck
        # Note there is an implicit timeout for barrier, which equal coordinator_pg_timeout
        dist.barrier(group=self.coordinator_process_group)

    @metrics.profile("torchelastic")
    def init_process_group(self):
        self.monitor_progress_step = 0
        dist.init_process_group(
            self.c10d_backend,
            timeout=timedelta(milliseconds=self.process_group_timeout),
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
        )

        if self.c10d_backend == dist.Backend.GLOO:
            self.coordinator_process_group = dist.group.WORLD
        else:
            # We don't need to use NCCL process group for control plane
            # collective operations, this helps us simplify our code (no need
            # to make it portable with NCCL)
            self.coordinator_process_group = dist.new_group(
                backend=dist.distributed_c10d.Backend.GLOO,
                timeout=timedelta(milliseconds=self.coordinator_pg_timeout),
            )

        log.info(
            "Initialized process group rank {0}, world size {1}".format(
                self.rank, self.world_size
            )
        )

    @metrics.profile("torchelastic")
    def should_save_checkpoint(self):
        """
        Whether the PET training loop need to do checkpoint.
        This normally happens when the job was explicitly ask for checkpoint.
        eg: executor got a preemption from scheduler
        """
        return False

    @metrics.profile("torchelastic")
    def should_rendezvous(self, state):
        if dist.get_world_size() == self.max_num_trainers:
            return False

        # Check if there are any new workers waiting at the rendezvous barrier
        num_new_nodes = torch.LongTensor([self.rendezvous.num_nodes_waiting()])

        # Use the GLOO based coordinator_process_group to perform the
        # collective op as we don't want to transfer these back-and-forth
        # between GPU and CPU (when GPUs are available).
        dist.all_reduce(
            num_new_nodes, op=dist.ReduceOp.MAX, group=self.coordinator_process_group
        )

        if num_new_nodes > 0:
            log.info(
                "Rank {0} detected {1} new nodes; will re-rendezvous.".format(
                    self.rank, num_new_nodes[0]
                )
            )
            return True
        else:
            return False

    @metrics.profile("torchelastic")
    def should_stop_training(self):
        # Check if coordinator wants the training to stop
        return self.stop_training

    @metrics.profile("torchelastic")
    def signal_training_done(self):
        # Close the rendezvous to indicate termination of the overall execution.
        # This also propagates the stop signal to other trainers.
        self.rendezvous.set_closed()
        self._destroy_process_group()
        self.stop_training = True

    @metrics.profile("torchelastic")
    def monitor_progress(self, state, worker_stats):
        self.monitor_progress_step += 1
        if (self.monitor_progress_step % self.MONITOR_PROGRESS_FREQ) != 0:
            return
        # In P2P, workers exchange progress rate info, and everyone compares itself
        # to the best worker in the group.

        # Logging only, no enforcement (T42935591)

        # All workers must participate in the collective communication, even if some
        # of them don't have a non-null WorkerStats or progress rate.
        if worker_stats is not None and worker_stats.get_progress_rate() is not None:
            prog_rate = worker_stats.get_progress_rate()
            prog_rate_known = True
        else:
            prog_rate = 0.0
            prog_rate_known = False

        gather_input = torch.FloatTensor([prog_rate, float(prog_rate_known)])
        gather_output = [torch.zeros_like(gather_input) for _ in range(self.world_size)]
        # use the GLOO based coordinator_process_group to perform the
        # collective op as we don't want to transfer these back-and-forth
        # between GPU and CPU (when GPUs are available).
        torch.distributed.all_gather(
            gather_output, gather_input, group=self.coordinator_process_group
        )

        if not prog_rate_known:
            # We don't know our own progress rate.
            return

        known_prog_rates = [val[0] for val in gather_output if val[1] > 0.0]
        if len(known_prog_rates) == 0:
            # No peer-trainer reported a progress rate
            return

        best_prog_rate = max(known_prog_rates)
        self.last_relative_prog_rate = float(prog_rate / best_prog_rate)

        # See: https://github.com/pytorch/elastic/issues/10
        straggler_threshold = 0.8
        if self.last_relative_prog_rate < straggler_threshold:
            self.is_worker_straggler = True
            log.warning(
                f"Straggler monitor: rank {self.rank} is slow "
                f"with relative performance of {self.last_relative_prog_rate} "
                f"(threshold: {straggler_threshold})"
            )
        else:
            self.is_worker_straggler = False

    @metrics.profile("torchelastic")
    def report_progress(self, state):
        pass

    @metrics.profile("torchelastic")
    def on_error(self, e):
        self._log_event("train_step_runtime_error", {"error": str(e)})
        log.error(
            "Rank: {0}\n"
            "Error: {1}\n"
            "ErrorType: {2}\n"
            "StackTrace: {3}".format(self.rank, str(e), type(e), traceback.format_exc())
        )
