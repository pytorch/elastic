#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import random
import sys
import threading
import time
from base64 import b64decode, b64encode
from typing import Optional

import etcd
from torch.distributed import Store, TCPStore, register_rendezvous_handler
from torchelastic.rendezvous import (
    RendezvousClosedException,
    RendezvousHandler,
    RendezvousNonRetryableError,
    RendezvousTimeoutException,
)


_log_fmt = logging.Formatter("%(levelname)s %(asctime)s %(message)s")
_log_handler = logging.StreamHandler(sys.stderr)
_log_handler.setFormatter(_log_fmt)

log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
log.addHandler(_log_handler)


# Retryable failure exception means the we were too late to make
# a desired state transition (e.g. because of a race condition),
# and should now restart from the beginning.
# A small delay is recommended to avoid spamming Etcd.
class EtcdRendezvousRetryableFailure(Exception):
    pass


# Similar to retryable failure, but the new state we observed suggests we
# can re-try immediately, i.e. without a need for "safety delay".
class EtcdRendezvousRetryImmediately(Exception):
    pass


# Default overall timeout for rendezvous barrier.
CONST_DEFAULT_OVERALL_TIMEOUT = 600

# Additional waiting amount after reaching num_min_workers,
# for the case rendezvous is elastic (min != max):
CONST_DEFAULT_LAST_CALL_TIMEOUT = 30

# Various constants used internally in EtcdRendezvous
CONST_ETCD_SETUP_TTL = 5
CONST_ETCD_FROZEN_TTL = 10
CONST_ETCD_JOINABLE_EPHEMERAL_TTL = 10

# Ephemeral node TTL for worker's keep-alive key:
CONST_WORKER_KEEPALIVE_TTL = 10

# TTL for the ephemeral run_id-specific directory. All rendezvous state data
# for a specific run_id (job instance) is contained within directory.
# Its only role is to clean-up rendezvous data from old runs (for the case when
# etcd server is persistent), and has no affect on correctnes, but should be
# larger than any timeouts that a worker process is expected to survive:
CONST_RUNID_SUBROOT_TTL = 7200  # 2 hours


# Delay (sleep) for a small random amount to reduce CAS failures.
# This does not affect correctness, but will reduce requests to etcd server.
def cas_delay():
    time.sleep(random.uniform(0, 0.1))


#
# EtcdRendezvousHandler implements a torchelastic.rendezvous.RendezvousHandler.
# It hides much of the implementation details, which are in EtcdRendezvous.
#
class EtcdRendezvousHandler(RendezvousHandler):
    def __init__(self, rdzv_impl):
        self._rdzv_impl = rdzv_impl

    def __del__(self):
        # TODO: look into using weakref here instead.
        del self._rdzv_impl

    def next_rendezvous(self):
        rdzv_version, rank, world_size = self._rdzv_impl.rendezvous_barrier()

        log.info("Creating EtcdStore as the c10d::Store implementation")
        store = self._rdzv_impl.setup_kv_store(rdzv_version)

        return store, rank, world_size

    def is_closed(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            return state["status"] == "closed"
        except etcd.EtcdKeyNotFound:
            # No rendezvous state, so it cannot be closed.
            return False

    def set_closed(self):
        self._rdzv_impl.set_closed()

    def num_nodes_waiting(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            if state["status"] == "final":
                return state["num_workers_waiting"]
        except etcd.EtcdKeyNotFound:
            pass
        return 0


# TODO: we should probably handle a few additional errors,
# like EtcdLeaderElectionInProgress and EtcdWatcherCleared. These are
# only relevant for multi-node Etcd ensemble. A simple retry would work,
# but is verbose to add everywhere. Consider wrapping the client calls
# into auto-retry for these errors?
#
class EtcdRendezvous(object):
    def __init__(
        self,
        endpoints,
        prefix,
        run_id,
        num_min_workers,
        num_max_workers,
        timeout,
        last_call_timeout,
        kwargs,
    ):
        self._prefix = prefix
        self._run_id = run_id
        self._num_min_workers = num_min_workers
        self._num_max_workers = num_max_workers
        self._timeout = timeout
        self._last_call_timeout = last_call_timeout

        # For cleaning up TTL refresher threads (for ephemeral keys)
        self._lease_run_id_stop = None
        self._lease_this_rank_stop = None

        if not self._prefix.endswith("/"):
            self._prefix += "/"

        self.client = etcd.Client(host=endpoints, allow_reconnect=True, **kwargs)
        log.info("Etcd machines: " + str(self.client.machines))

        # Setup a permanent prefix dir, if didn't exist
        if self._prefix != "/":
            self.create_path_if_not_exists(self._prefix)

        # Lease a "sub-root" node specific to this job instance (run_id)
        self.create_path_if_not_exists(self.get_path(""), ttl=CONST_RUNID_SUBROOT_TTL)
        self._lease_run_id_stop = self.setup_lease_renewal(
            self.get_path(""), ttl=CONST_RUNID_SUBROOT_TTL
        )

        # Subdir for all rendezvous work
        self.create_path_if_not_exists(self.get_path("/rdzv"))

        # Create a rendezvous version counter, if doesn't exist
        try:
            self.client.write(
                key=self.get_path("/rdzv/version_counter"), value="0", prevExist=False
            )
        except etcd.EtcdAlreadyExist:
            pass

    def __del__(self):
        # TODO: look into using weakref here instead.
        if self._lease_run_id_stop is not None:
            self._lease_run_id_stop.set()

        if self._lease_this_rank_stop is not None:
            self._lease_this_rank_stop.set()

    #
    # Main entry point for next rendezvous.
    # This method is blocking until rendezvous succeeds or a timeout occurs.
    #
    def rendezvous_barrier(self):
        self._rendezvous_deadline = time.time() + self._timeout
        while True:
            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutException()

            log.info("Attempting to join next rendezvous")
            try:
                # Dis-own our lease in the previous rendezvous, if exists
                if self._lease_this_rank_stop is not None:
                    self._lease_this_rank_stop.set()

                return self.init_phase()

            except EtcdRendezvousRetryImmediately:
                # The type of failure suggests we can retry without delay
                pass

            except EtcdRendezvousRetryableFailure:
                # In case of retryable failure, wait a small delay
                # to avoid spamming etcd
                time.sleep(1)

            except RendezvousTimeoutException:
                log.info("Rendezvous timeout occured in EtcdRendezvousHandler")
                raise

            except RendezvousClosedException:
                log.info(
                    f"Rendezvous for run_id={self._run_id} was observed to be closed"
                )
                raise

            except RendezvousNonRetryableError:
                raise

            except Exception as e:
                # In case of a general exception, wait a small delay
                # to avoid spamming etcd
                # FIXME: there are a few things that fall under this like
                # etcd.EtcdKeyNotFound, etc, which could be handled more explicitly.
                log.info("Rendezvous attempt failed, will retry. Reason: " + str(e))
                time.sleep(1)

    #
    # Initially, the rendezvous state is expected to be one of:
    #  a) empty (non-existent) - in this case we try to create a new one.
    #  b) joinable - we try to join it.
    #  c) final - we announce ourselves as waiting, and go into monitoring mode
    #             for this rendezvous.
    #
    # Any other state is considered transitional, and will be retried after
    # a short delay.
    #
    def init_phase(self):
        try:
            active_version = self.try_create_rendezvous()
            state = json.loads(active_version.value)
            log.info("New rendezvous state created: " + str(state))
        except etcd.EtcdAlreadyExist:
            active_version, state = self.get_rdzv_state()
            # Note: it is possible for above query to fail (etcd.EtcdKeyNotFound),
            # but this is ok for us - just means we'll restart from beginning.
            log.info("Observed existing rendezvous state: " + str(state))

        if state["status"] == "closed":
            raise RendezvousClosedException()

        if state["status"] == "joinable":
            return self.join_phase(state["version"])

        if state["status"] == "final":
            self.handle_existing_rendezvous(state["version"])
            raise EtcdRendezvousRetryImmediately()

        # We observed some intermediate state, which is best handled by retrying later
        self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
        raise EtcdRendezvousRetryableFailure()

    #
    # We observed a rendezvous state in 'joinable' state, and attempt to join this
    # particular version, and then wait for all other peers to join.
    #
    def join_phase(self, expected_version):
        # Failure to join will propagate an exception, causing a re-entry.
        active_version, this_rank = self.join_rendezvous(expected_version)
        state = json.loads(active_version.value)
        log.info(
            "Joined rendezvous version {} as rank {}. Full state: {}".format(
                state["version"], this_rank, state
            )
        )

        # If this worker was first to reach num_min_workers requirement,
        # and rendezvous is still joinable (therefore it is elastic),
        # then this worker will be repsonsible for waiting out the "last call"
        # timeout and closing (i.e. transitioning to 'frozen') the rendezvous
        # afterwards.
        # As a safety against a potential failure of this worker (during the
        # last call timeout), the rendezvous state is made ephemeral
        # when min_num_workers is reached.

        if this_rank == self._num_min_workers - 1 and state["status"] == "joinable":
            log.info("Rank {} is responsible for join last call.".format(this_rank))
            last_call_deadline = time.time() + self._last_call_timeout
            self.handle_join_last_call(expected_version, last_call_deadline)
            log.info("Rank {} finished join last call.".format(this_rank))

        # Wait for rendezvous state to be frozen, which means a fixed set of peers
        log.info("Waiting for remaining peers.")
        active_version = self.wait_for_peers(expected_version)
        state = json.loads(active_version.value)

        assert (
            state["version"] == expected_version
        ), "Logic error: failed to observe version mismatch"

        return self.confirm_phase(expected_version, this_rank)

    #
    # Once the rendezvous state transitiosn from 'joinable' to 'frozen',
    # we have every participant confirm their membership and setup per-member
    # keep-alive TTL keys, and then wait for all other participants to confirm,
    # which would then successfully conclude this rendezvous.
    #
    def confirm_phase(self, expected_version, this_rank):
        log.info("All peers arrived. Confirming membership.")
        self.confirm_membership(expected_version, this_rank)

        log.info("Waiting for confirmations from all peers.")
        active_version = self.wait_for_final(expected_version)
        state = json.loads(active_version.value)

        log.info(
            "Rendezvous version {} is complete. Final state: {}".format(
                state["version"], state
            )
        )

        # Rendezvous version number; our rank in it; world size
        return state["version"], this_rank, len(state["participants"])

    #
    # Handle the case when there's an existing (state 'final) rendezvous already
    # in place, and we have to announce ourselves waiting, and wait until
    # the next rendezvous opportunity.
    #
    def handle_existing_rendezvous(self, expected_version):
        # If statie is 'final' -> increment num_workers_waiting
        # Then, observe state changes:
        #   1. if it's no longer final -> bail out and re-try
        #   2. if keep alives are missing, destroy it and bail out.
        active_state = self.announce_self_waiting(expected_version)
        log.info(
            "Added self to waiting list. Rendezvous full state: {}".format(
                active_state.value
            )
        )

        self.wait_for_rendezvous_to_free(expected_version)
        log.info("Previously existing rendezvous state changed. Will re-try joining.")

    # Create new rendezvous state, or raise etcd.EtcdAlreadyExist exception.
    def try_create_rendezvous(self):
        # Initially active_version is ephemeral - this is to handle the
        # possibility that might fail to complete the setup transaction,
        # i.e. the transition "setup" -> "joinable".
        active_version = self.client.write(
            key=self.get_path("/rdzv/active_version"),
            value=json.dumps({"status": "setup"}),
            prevExist=False,
            ttl=CONST_ETCD_SETUP_TTL,
        )

        try:
            version_counter = self.client.get(self.get_path("/rdzv/version_counter"))
            version_counter.value = str(int(version_counter.value) + 1)
            self.client.update(version_counter)
        except (etcd.EtcdKeyNotFound, etcd.EtcdCompareFailed):
            raise RendezvousNonRetryableError(
                "Unexpected state of EtcdRendezvousHandler, worker needs to die."
            )

        # Any failure below results in declaring a retryable rendezvous failure.
        # The ephemeral /rdzv/active_version will expire and someone can then
        # re-try the setup process.

        # Create directory node for participant data
        self.client.write(
            key=self.get_path("/rdzv/v_{}".format(version_counter.value)),
            value=None,
            dir=True,
            prevExist=False,
        )

        # Publish rendezvous version and signal it is ready-to-be-joined.
        # If rendezvous was set closed just before this, a retry will happen,
        # where the closed condition will be handled.
        return self.client.test_and_set(
            key=self.get_path("/rdzv/active_version"),
            value=json.dumps(
                {
                    "status": "joinable",
                    "version": version_counter.value,
                    "participants": [],
                }
            ),
            prev_value=active_version.value,
        )

    # Helper method for the join phase.
    def join_rendezvous(self, expected_version):
        # Use compare-and-swap to add self to rendezvous state:
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()

            if state["status"] != "joinable":
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state became non-joinable before we could join. "
                    "Must join next one."
                )

            if state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous version changed. Must try join the new one."
                )

            assert (
                len(state["participants"]) < self._num_max_workers
            ), "Logic error: joinable rendezvous should always have space left"

            this_rank = len(state["participants"])
            state["participants"].append(this_rank)

            # When reaching min workers, or changing state to frozen, we'll set
            # the active_version node to be ephemeral.
            if len(state["participants"]) == self._num_max_workers:
                state["status"] = "frozen"
                state["keep_alives"] = []
                set_ttl = CONST_ETCD_FROZEN_TTL
            elif len(state["participants"]) >= self._num_min_workers:
                set_ttl = CONST_ETCD_JOINABLE_EPHEMERAL_TTL
            else:
                set_ttl = None

            try:
                # Compare-and-swap.
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                    ttl=set_ttl,
                )
                # We succeeded joining.
                return active_version, this_rank

            except etcd.EtcdCompareFailed:
                log.info("Join rendezvous CAS unsuccessful, retrying")

    # Helper method for the join phase.
    def wait_for_peers(self, expected_version):
        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] == "frozen" and state["version"] == expected_version:
                # Success, all peers arrived.
                return active_version

            elif state["status"] == "joinable" and state["version"] == expected_version:
                # Continue waiting for any interesting events.
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1
                )

            else:
                # No valid transition possible at this point
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )

    # Helper method for the confirm phase
    def confirm_membership(self, expected_version, this_rank):
        # Compare-and-swap loop
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()

            if state["status"] != "frozen":
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous no longer frozen, before we confirmed. "
                    "Must join next one"
                )
            if state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous version changed. Must try join the new one."
                )

            this_lease_key = self.get_path(
                "/rdzv/v_{}/rank_{}".format(expected_version, this_rank)
            )
            self.client.set(this_lease_key, value=None, ttl=CONST_WORKER_KEEPALIVE_TTL)

            state["keep_alives"].append(this_lease_key)
            if len(state["keep_alives"]) == len(state["participants"]):
                # Everyone confirmed (this rank is last to do so)
                state["status"] = "final"
                state["num_workers_waiting"] = 0
                finalize = True
            else:
                finalize = False

            try:
                # Compare-and-swap. If new state is still frozen, keep it ephemeral.
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                    ttl=None if finalize else CONST_ETCD_FROZEN_TTL,
                )

                self._lease_this_rank_stop = self.setup_lease_renewal(
                    this_lease_key, ttl=CONST_WORKER_KEEPALIVE_TTL
                )
                return active_version

            except etcd.EtcdCompareFailed:
                log.info("Confirm membership CAS unsuccessful, retrying")

    # Helper method for the confirm phase
    def wait_for_final(self, expected_version):
        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] == "final" and state["version"] == expected_version:
                # Succcess. This rendezvous is final, and we accept it.
                return active_version

            elif state["status"] == "frozen" and state["version"] == expected_version:
                # Continue waiting for any interesting events.
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1
                )

            else:
                # No valid transition possible at this point
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )

    # Announce this worker is waiting (via num_workers_waiting counter) to join next
    # rendezvous, but only if state and version match.
    def announce_self_waiting(self, expected_version):
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()

            if state["status"] != "final" or state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately()

            # Increment counter to signal an additional waiting worker.
            state["num_workers_waiting"] += 1

            try:
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                )
                return active_version

            except etcd.EtcdCompareFailed:
                log.info("Announce self as waiting CAS unsuccessful, retrying")

    #
    # When there's an existing valid rendezvous in state 'final', we have to wait until
    # the next opportunity to join.
    #
    # Such opportunity may come from:
    #  a) rendezvous state changed by someone else, in which case we unblock and retry.
    #  b) rendezvous becomes invalid because at least one member failed to renew their
    #     leased keep_alive node. We detect this, and destroy the rendezvous.
    #
    def wait_for_rendezvous_to_free(self, expected_version):
        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] != "final" or state["version"] != expected_version:
                return

            # Check if current rendezvous state is valid, in the sense that all
            # its members are alive (renewing their lease).
            # If not, try destroy this rendezvous, so a new one can be created.
            alive_members = self.client.get(
                self.get_path("/rdzv/v_{version}".format(version=expected_version))
            )
            keep_alive_keys = [ch.key for ch in alive_members.children]

            for key in state["keep_alives"]:
                if key not in keep_alive_keys:
                    # This participant didn't renew their lease. We'll declare this
                    # rendezvous version as dead (but only if it hadn't changed)
                    log.info("Keep-alive key {} is not renewed.".format(key))
                    log.info(
                        "Rendevous version {} is incomplete. ".format(expected_version)
                    )
                    log.info("Attempting to destroy it.")

                    # Compare-and-delete operation. Throws if compare failed,
                    # which means rendezvous was already destroyed/re-created/closed,
                    # and we can try to re-enter the barrier.
                    self.client.delete(
                        key=self.get_path("/rdzv/active_version"),
                        prevValue=active_version.value,
                    )

                    log.info(
                        "Destroyed rendezvous version {} successfully.".format(
                            expected_version
                        )
                    )

                    # We can return (and retry) immediately
                    return

            # Existing rendezvous seems valid, no reason to destroy it.
            # We just have to wait until something changes and re-check.
            try:
                overall_timeout = (
                    max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
                )
                self.client.watch(
                    key=self.get_path("/rdzv"),
                    index=active_version.etcd_index + 1,
                    recursive=True,
                    timeout=overall_timeout,
                )
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass

            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutException()
            active_version, state = self.get_rdzv_state()

    #
    # After we reach min number of workers, one particular worker takes on the
    # responsibility of waiting an additional timeout before closing the join window.
    # If the worker responsible for this fails, the rendezvous will be destroyed due
    # to expiring TTL, and the other participants will re-rendezvous.
    #
    def handle_join_last_call(self, expected_version, deadline):
        # Here we expect to see state <joinable, expected_version>
        # Exit gracefully if either:
        #  1. state becomes <frozen, expected_version>
        #  2. timeout happens (reaching deadline), in which case
        #     we try the tranisiton to <frozen, expected_version>
        # Exit with exception otherwise.

        active_version, state = self.get_rdzv_state()
        while True:
            if state["status"] == "frozen" and state["version"] == expected_version:
                # Worker set became frozen before last-call timeout. This is possible
                # when num_max_workers is reached before the tiemout.
                return

            if state["status"] != "joinable" or state["version"] != expected_version:
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )

            # If timeout occurred, attempt a state transition (joinable -> frozen)
            if time.time() >= deadline:
                state["status"] = "frozen"
                state["keep_alives"] = []
                try:
                    active_version = self.client.test_and_set(
                        key=self.get_path("/rdzv/active_version"),
                        value=json.dumps(state),
                        prev_value=active_version.value,
                        ttl=CONST_ETCD_FROZEN_TTL,
                    )
                    # We successfully made this rendezvous frozen.
                    return
                except etcd.EtcdCompareFailed:
                    log.info("Join last-call transition CAS unsuccessful. Will retry")
                    cas_delay()
                    active_version, state = self.get_rdzv_state()
                    continue

            # Timeout did not occur, so we must refresh TTL, and wait for
            # further changes. Note: we only want TTL to be refreshed if
            # state is still joinable, hence we use CAS for that here,
            # even though we don't change any of the data.
            try:
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=active_version.value,
                    prev_value=active_version.value,
                    ttl=CONST_ETCD_JOINABLE_EPHEMERAL_TTL,
                )

                # Minimize "oversleeping":
                timeout = min(
                    CONST_ETCD_JOINABLE_EPHEMERAL_TTL / 2,
                    deadline - time.time() + 1.0,  # Oversleeping by 1s is ok.
                )
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1, timeout=timeout
                )
            except etcd.EtcdCompareFailed:
                log.info("Join last-call TTL refresh CAS unsuccessful, will retry")
                cas_delay()
                active_version, state = self.get_rdzv_state()

    # Mark rendezvous 'closed' for current run_id, which is used to signal other
    # participants to not attempt to perform (re-)rendezvous. This is useful
    # when one of the workers decides the job is complete.
    def set_closed(self):
        while True:
            active_version, state = self.get_rdzv_state()

            if state["status"] == "closed":
                # Already closed by someone else.
                return

            state["status"] = "closed"
            try:
                self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                )
                return

            except etcd.EtcdCompareFailed:
                log.info("Set closed CAS unsuccessful, retrying")
                cas_delay()

    def get_rdzv_state(self):
        active_version = self.client.get(key=self.get_path("/rdzv/active_version"))
        return active_version, json.loads(active_version.value)

    def try_wait_for_state_change(self, etcd_index, timeout=None):
        # Don't sleep past the overall deadline (at least more than by 1s)
        overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
        timeout = overall_timeout if timeout is None else min(timeout, overall_timeout)

        try:
            self.client.watch(
                self.get_path("/rdzv/active_version"), index=etcd_index, timeout=timeout
            )
        except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
            pass

        if time.time() > self._rendezvous_deadline:
            raise RendezvousTimeoutException()

        # Unfortunately, we have to do another fetch in order to get last etcd_index.
        return self.get_rdzv_state()

    def get_path(self, path):
        if not path.startswith("/"):
            path = "/" + path

        return "{prefix}run_{run_id}{path}".format(
            prefix=self._prefix, run_id=self._run_id, path=path
        )

    def create_path_if_not_exists(self, full_path, ttl=None):
        try:
            self.client.write(
                key=full_path, value=None, dir=True, prevExist=False, ttl=ttl
            )
        except etcd.EtcdAlreadyExist:
            pass

    def setup_lease_renewal(self, full_path, ttl):
        # NOTE: For ephemeral key TTL renewal (~lease) to work correctly,
        # make sure you don't call any long-blocking methods that do not
        # release the Python's GIL! An example of this is calling a pybind11
        # extension function that is blocking / long-running, but is not
        # doing a scoped release of the GIL.
        def lease_worker(client, path, ttl, stop_event):
            while True:
                try:
                    client.refresh(path, ttl=ttl)
                except etcd.EtcdKeyNotFound:
                    break

                if stop_event.wait(timeout=ttl / 2):
                    break

        lease_stop_event = threading.Event()
        lease_thread = threading.Thread(
            target=lease_worker, args=(self.client, full_path, ttl, lease_stop_event)
        )

        lease_thread.daemon = True
        lease_thread.start()

        return lease_stop_event

    def store_extra_data(self, rdzv_version, key, value):
        node = self.get_path("/rdzv/v_{}/extra_data".format(rdzv_version))
        try:
            # If first time we are storing anything:
            extra_data = self.client.write(
                key=node, value=json.dumps({key: value}), prevExist=False
            )
            return
        except etcd.EtcdAlreadyExist:
            pass

        # CAS loop, to make sure we don't lose concurrent stores.
        while True:
            # We never delete extra_data. Failure here should be fatal, no special handling.
            extra_data = self.client.get(node)

            new_extra_data_value = json.loads(extra_data.value)
            new_extra_data_value[key] = value

            try:
                extra_data = self.client.test_and_set(
                    key=node,
                    value=json.dumps(new_extra_data_value),
                    prev_value=extra_data.value,
                )
                return
            except etcd.EtcdCompareFailed:
                log.info("Store extra_data CAS unsuccessful, retrying")
                time.sleep(0.1)

    def load_extra_data(self, rdzv_version, key, timeout=None):
        # 'extra_data' node itself, and the directory it is located in:
        node = self.get_path("/rdzv/v_{}/extra_data".format(rdzv_version))
        node_dir = self.get_path("/rdzv/v_{}".format(rdzv_version))

        # TODO: implement timeout
        # https://github.com/pytorch/elastic/issues/12
        while True:
            # Combined wait for the node itself, and the key inside it.
            root = self.client.get(node_dir)

            # Find the extra_data node, if it exists
            extra_data = [n for n in root.children if n.key == node]
            assert len(extra_data) <= 1

            # Node for extra_data exists, check the desired key inside it.
            if len(extra_data) == 1:
                extra_data_dict = json.loads(extra_data[0].value)
                if key in extra_data_dict:
                    return extra_data_dict[key]

            # The 'extra_data' node doesn't exist, or they key isn't published yet.
            # Wait for interesting events on the extra_data node and retry.
            try:
                self.client.watch(node, index=root.etcd_index + 1)
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass

    def setup_kv_store(self, rdzv_version):
        store_path = self.get_path(f"/rdzv/v_{rdzv_version}/kv")
        self.create_path_if_not_exists(store_path)
        return EtcdStore(etcd_client=self.client, etcd_store_prefix=store_path)


class EtcdStore(Store):
    """
    Implements a c10 Store interface by piggybacking on the rendezvous etcd instance.
    """

    def __init__(
        self,
        etcd_client,
        etcd_store_prefix,
        timeout: Optional[datetime.timedelta] = None,
    ):
        super().__init__()  # required for pybind trampoline.

        self.client = etcd_client
        self.prefix = etcd_store_prefix

        # Default timeout same as in c10d/Store.hpp
        self.timeout = (
            timeout if timeout is not None else datetime.timedelta(seconds=300)
        )
        if not self.prefix.endswith("/"):
            self.prefix += "/"

    def set(self, key, value):
        """
        Write a key/value pair into EtcdStore.
        Both key and value may be either Python str or bytes.
        """
        self.client.set(key=self.prefix + self._encode(key), value=self._encode(value))

    def get(self, key) -> bytes:
        """
        Get a value by key, possibly doing a blocking wait.

        If key is not immediately present, will do a blocking wait
        for at most `timeout` duration or until the key is published.
        If key still not published after timeout, a LookupError is raised.

        Return value is of type bytes.
        """
        b64_key = self.prefix + self._encode(key)
        kvs = self._try_wait_get([b64_key])

        if kvs is None:
            raise LookupError(f"Key {key} not found in EtcdStore")

        return self._decode(kvs[b64_key])

    def add(self, key, num: int) -> int:
        """
        Atomically increment a value by an integer amount, returns the new value.
        Returned value is of type `int`.

        The integer is represented as a string using base 10.
        If key is not present, a default value of "0" will be assumed.
        """
        b64_key = self._encode(key)
        # c10d Store assumes value is an integer represented as a decimal string
        try:
            # Assume default value "0", if this key didn't yet:
            node = self.client.write(
                key=self.prefix + b64_key,
                value=self._encode(str(num)),  # i.e. 0 + num
                prevExist=False,
            )
            return int(self._decode(node.value))
        except etcd.EtcdAlreadyExist:
            pass

        while True:
            # Note: c10d Store does not have a method to delete keys, so we
            # can be sure it's still there.
            node = self.client.get(key=self.prefix + b64_key)
            new_value = self._encode(str(int(self._decode(node.value)) + num))
            try:
                node = self.client.test_and_set(
                    key=node.key, value=new_value, prev_value=node.value
                )
                return int(self._decode(node.value))
            except etcd.EtcdCompareFailed:
                cas_delay()

    def wait(self, keys, override_timeout: Optional[datetime.timedelta] = None):
        """
        Waits until all of the keys are published, or until timeout.
        If timeout occurs first, a LookupError is raied.
        """
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout)
        if kvs is None:
            raise LookupError("Timeout while waiting for keys in EtcdStore")
        # No return value on success

    def check(self, keys) -> bool:
        """
        Check if all of the keys are immediately present (without waiting).
        Returns bool.
        """
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(
            b64_keys,
            override_timeout=datetime.timedelta(microseconds=1),  # as if no wait
        )
        return kvs is not None

    def set_timeout(self, timeout: datetime.timedelta):
        """
        Change the timeout used for all future operations.
        """
        self.timeout = timeout

    #
    # Encode key/value data in base64, so we can store arbitrary binary data
    # in EtcdStore. Input can be `str` or `bytes`.
    # In case of `str`, utf-8 encoding is assumed.
    #
    def _encode(self, value) -> str:
        if type(value) == bytes:
            return b64encode(value).decode()
        elif type(value) == str:
            return b64encode(value.encode()).decode()
        raise ValueError("Value must be of type str or bytes")

    #
    # Decode a base64 string (of type `str` or `bytes`).
    # Return type is `bytes`, which is more convenient with the Store interface.
    #
    def _decode(self, value) -> bytes:
        if type(value) == bytes:
            return b64decode(value)
        elif type(value) == str:
            return b64decode(value.encode())
        raise ValueError("Value must be of type str or bytes")

    #
    # Get all of the (base64-encoded) etcd keys at once, or wait until all the keys
    # are published or timeout occurs.
    # This is a helper method for the public interface methods.
    #
    # On success, a dictionary of {etcd key -> etcd value} is returned.
    # On timeout, None is returned.
    #
    def _try_wait_get(self, b64_keys, override_timeout=None):
        timeout = self.timeout if override_timeout is None else override_timeout
        deadline = time.time() + timeout.total_seconds()

        while True:
            # Read whole directory (of keys), filter only the ones waited for
            all_nodes = self.client.get(key=self.prefix)
            req_nodes = {
                node.key: node.value
                for node in all_nodes.children
                if node.key in b64_keys
            }

            if len(req_nodes) == len(b64_keys):
                # All keys are available
                return req_nodes

            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                return None

            try:
                self.client.watch(
                    key=self.prefix,
                    recursive=True,
                    timeout=watch_timeout,
                    index=all_nodes.etcd_index + 1,
                )
            except etcd.EtcdWatchTimedOut:
                if time.time() >= deadline:
                    return None
                else:
                    continue
            except etcd.EtcdEventIndexCleared:
                continue


def _get_socket_with_port():
    import socket

    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            log.info("Socket creation attempt failed: " + e)

    raise RuntimeError("Failed to create a socket")


# Helper for _etcd_rendezvous_handler(url)
def _parse_etcd_client_params(params):
    kwargs = {}
    if "protocol" in params:
        protocol = params["protocol"]
        assert protocol in ["http", "https"], "Protocol must be http or https."
        kwargs["protocol"] = protocol
    if "cacert" in params:
        kwargs["ca_cert"] = params["cacert"]
    if "cert" in params:
        if "key" in params:
            # python-etcd client expects key as a second element of `cert` tuple
            kwargs["cert"] = (params["cert"], params["key"])
        else:
            kwargs["cert"] = params["cert"]
    return kwargs


# Handler for torch.distributed "static" registration
def _etcd_rendezvous_handler(url):
    """
    Example URLs:
        etcd://localhost:2379/123?min_workers=4&max_workers=8&timeout=300
        etcd://192.168.0.42/123?etcd_prefix=/custom_prefix/foo&min_workers=4
        etcd://localhost:2379/123?min_workers=4&protocol=https&cacert=/etc/kubernetes/certs/ca.crt&cert=/etc/kubernetes/certs/client.crt&key=/etc/kubernetes/certs/client.key

    Where:
        123 - the run_id (unique id for this training job instance),
        min_workers=4 - min number of workers expected to join the rendezvous,
        max_workers=8 - max number of workers allowed to join the rendezvous,
                        defaults to min_workers is not specified.
        timeout=300 - total timeout within which next_rendezvous is expected to
                      succeed; a RendezvousTimeoutException is raised otherwise;
                      Defaults is 600 (10 minutes).
        last_call_timeout - additional wait amount ("last call") after
                            min number of workers has been reached.
                            Defaults to 30 seconds.
        etcd_prefix - path prefix (from etcd root), inside which all
                      etcd nodes will be created.
                      Default is "/torchelastic/p2p".
        protocol=https - http (default) or https to access etcd.
        cacert=/etc/kubernetes/certs/ca.crt - CA cert to access etcd,
                    only makes sense with https.
        cert=/etc/kubernetes/certs/client.crt - client cert to access etcd,
                    only makes sense with https.
        key=/etc/kubernetes/certs/client.key - client key to access etcd,
                    only makes sense with https.

    """
    import re
    from urllib.parse import urlparse

    url = urlparse(url)
    assert url.scheme == "etcd"

    # Etcd endpoints. (Current url format only allows a single host)
    endpoint = url.netloc
    match = re.match(r"(.+):(\d+)$", endpoint)  # check if port was provided
    if match:
        etcd_endpoints = ((match.group(1), int(match.group(2))),)
    else:
        # Use default etcd port
        etcd_endpoints = ((endpoint, 2379),)

    # Run ID value -> unique identifier of this training job instance:
    # typically a job_id or name assigned by the scheduler or user
    run_id = url.path.strip("/")

    # Parse all of query parameters:
    params = dict(pair.split("=") for pair in filter(None, url.query.split("&")))

    etcd_prefix = params.get("etcd_prefix", "/torchelastic/p2p")
    num_min_workers = int(params["min_workers"])
    num_max_workers = int(params.get("max_workers", num_min_workers))
    assert num_min_workers >= 1, "Min number of workers should be at least 1"
    assert (
        num_max_workers >= num_min_workers
    ), "Max number of workers cannot be less than min number of workers"

    timeout = int(params.get("timeout", CONST_DEFAULT_OVERALL_TIMEOUT))
    last_call_timeout = int(
        params.get("last_call_timeout", CONST_DEFAULT_LAST_CALL_TIMEOUT)
    )

    kwargs = _parse_etcd_client_params(params)

    # Etcd rendezvous implementation
    etcd_rdzv = EtcdRendezvous(
        endpoints=etcd_endpoints,
        prefix=etcd_prefix,
        run_id=run_id,
        num_min_workers=num_min_workers,
        num_max_workers=num_max_workers,
        timeout=timeout,
        last_call_timeout=last_call_timeout,
        kwargs=kwargs,
    )
    return EtcdRendezvousHandler(rdzv_impl=etcd_rdzv)


# torchelastic.rendezvous.RendezvousHandler using etcd (API v2):
register_rendezvous_handler("etcd", _etcd_rendezvous_handler)
