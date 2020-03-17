#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import multiprocessing
import os
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torchelastic.train_loop as elastic_train_loop
from test_mocks import (
    RadixTestDataset,
    TestCoordinatorP2P,
    TestDataset,
    TestState,
    TestStateFailOnSync,
    TestStateWithRollbackDisabled,
    TestWorkerStats,
    test_checkpoint_manager,
)
from test_utils import TestCommon, _get_or_raise
from torchelastic.checkpoint import FileSystemCheckpointManager
from torchelastic.coordinator import NonRetryableException
from torchelastic.p2p.coordinator_p2p import CoordinatorP2P


# _train_step only runs hooks on this step and later
_RUN_HOOKS_AFTER_STEP = 3

log = logging.getLogger(__name__)


def _train_step(state, hooks):
    """
    This is the common training function used by all unit tests in this file.
    It simply grabs a value from the data iterator and appends it to an array
    in `state`.
    Starting on iteration `_RUN_HOOKS_AFTER_STEP`, this function will run
    any caller-provided hook functions to trigger failures.
    """
    if "steps" not in _train_step.__dict__:
        _train_step.steps = 0

    # Record worker rank, so we can write deterministic rank-dependent tests
    state.set_worker_rank(dist.get_rank())

    # Invariant necessary for rank-dependent tests to be valid:
    assert state.dataset.rank == state.get_worker_rank()

    sample = next(state.get_data_iterator())

    # Perform all-reduce as a proxy for synchronous sgd
    tensor = torch.tensor([sample])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    state.total_sum += int(tensor[0])
    log.info(
        f"After all_reduce: rank {state.get_worker_rank()}, "
        f"sample {sample}, "
        f"PID: {os.getpid()}"
        f"train step sum: {int(tensor[0])}, total sum: {state.total_sum}"
    )
    state.nums.append(sample)

    # Only run hooks after the Nth iteration.
    # Failing here means that we roll back 1 iteration and thus
    # only N-1 iterations remain committed to the shared state
    _train_step.steps += 1
    if _train_step.steps >= _RUN_HOOKS_AFTER_STEP:
        if hooks is not None:
            for _, hook in hooks.items():
                hook()

    # need a final barrier so that all processes move step # in lock-step
    # otherwise there is a race condition in rollbacks where process A
    # could be done with step n, committed its state, and moves onto step n+1
    # if process B throws an exception in step n, then the rollback in
    # process A starts at step n+1 rather than re-executing step n
    dist.barrier()

    return state


def _make_elastic_train_step(train_step, hooks, worker_stats=None):
    """
    Given an "application-domain" train_step, wrap it into
    an elastic_train_step which torchelastic train_loop can drive.
    """

    def elastic_train_step(state):
        return train_step(state, hooks), worker_stats

    return elastic_train_step


class ElasticTrainerTestBase(TestCommon, abc.ABC):
    """
    Defines the common set of end-to-end elastic training tests,
    allowing the derived class to inject a rendezvous url.
    Useful for testing different types of rendezvous algorithms.

    Usage: see etcd_elastic_trainer_test.py
    """

    def setUp(self):
        TestCommon.setUp(self)
        self.test_dir = tempfile.TemporaryDirectory()  # noqa

        self.min_size = 2
        self.max_size = 4

    def tearDown(self):
        TestCommon.tearDown(self)
        self.test_dir.cleanup()

    @abc.abstractmethod
    def get_rdzv_url(self, run_id, min_size, max_size):
        pass

    def _train_common(
        self,
        _,
        elastic_coordinator,
        train_step,
        hooks,
        state_override=None,
        worker_stats=None,
    ):
        state = TestState() if state_override is None else state_override

        elastic_train_step = _make_elastic_train_step(train_step, hooks, worker_stats)
        state = elastic_train_loop.train(elastic_coordinator, elastic_train_step, state)
        return state

    def _train(self, _, run_id, train_step, hooks, state_override=None, timeout=600):
        """
        Common sub-process trainer entry point used by most tests.
        """
        elastic_coordinator = CoordinatorP2P(
            c10d_backend="gloo",
            init_method=self.get_rdzv_url(
                run_id, self.min_size, self.max_size, timeout=timeout
            ),
            max_num_trainers=self.max_size,
            process_group_timeout=10000,
        )
        return self._train_common(
            _, elastic_coordinator, train_step, hooks, state_override
        )

    def _train_with_checkpoint(self, _, run_id, train_step, hooks, state_override=None):
        """
        Train with checkpoint loading/saving
        """
        with test_checkpoint_manager(self.test_dir.name):
            elastic_coordinator = TestCoordinatorP2P(
                c10d_backend="gloo",
                init_method=self.get_rdzv_url(run_id, self.min_size, self.max_size),
                max_num_trainers=self.max_size,
                process_group_timeout=10000,
            )
            state = self._train_common(
                _, elastic_coordinator, train_step, hooks, state_override
            )
            return state

    def _train_with_worker_stats(
        self,
        _,
        run_id,
        train_step,
        hooks,
        state_override=None,
        worker_stats_progress_rate=None,
    ):
        """
        Similar to `_train`, but uses a coordinator that validates WorkerStats object
        """
        fixed_worker_stats = TestWorkerStats(progress_rate=worker_stats_progress_rate)

        elastic_coordinator = CoordinatorP2P(
            c10d_backend="gloo",
            init_method=self.get_rdzv_url(run_id, self.min_size, self.max_size),
            max_num_trainers=self.max_size,
            process_group_timeout=10000,
        )
        return self._train_common(
            _,
            elastic_coordinator,
            train_step,
            hooks,
            state_override,
            fixed_worker_stats,
        )

    def _train_rerendezvous(self, _, run_id, train_step, hooks, state_override=None):
        """
        Alternate sub-process trainer entry point used by tests that want to
        force a re-rendezvous after every iteration.
        """

        class RerendezvousCoordinatorP2P(CoordinatorP2P):
            def should_rendezvous(self, state):
                return True

        elastic_coordinator = RerendezvousCoordinatorP2P(
            c10d_backend="gloo",
            init_method=self.get_rdzv_url(run_id, self.min_size, self.max_size),
            max_num_trainers=self.max_size,
            process_group_timeout=10000,
        )
        state = self._train_common(
            _, elastic_coordinator, train_step, hooks, state_override
        )
        return state

    def test_normal_flow(self):
        """
        Test a very simple 4 trainer case.
        """
        run_id = self._generate_run_id()

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None)
            qouts.append(qout)
            qerrs.append(qerr)

        # get the samples that each worker processed and assert against input data
        nums = {}
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            self.assertEqual(5, len(state.nums))
            nums[state.get_worker_rank()] = state.nums
            sums.append(state.total_sum)

        # All 4 trainers should train 5 samples without issue
        # (due to no re-rendezvous, we can assume rank-stability)
        self.assertEqual([11, 15, 19, 23, 27], nums[0])
        self.assertEqual([12, 16, 20, 24, 28], nums[1])
        self.assertEqual([13, 17, 21, 25, 29], nums[2])
        self.assertEqual([14, 18, 22, 26, 30], nums[3])
        self.assertEqual([410, 410, 410, 410], sums)  # 410 = 11 + 12 + ... + 30

    def test_rdzv_timeout(self):
        """
        Test timeout exception.
        """
        run_id = self._generate_run_id()

        nprocs = 4
        self.min_size = nprocs
        qouts = []
        qerrs = []
        timeout = 30
        for _ in range(0, nprocs - 1):
            _, qout, qerr = self._spawn(
                self._train, run_id, _train_step, None, None, timeout
            )
            qouts.append(qout)
            qerrs.append(qerr)

        # get the samples that each worker processed and assert against input data
        for i in range(0, nprocs - 1):
            with self.assertRaises(NonRetryableException) as err:
                _get_or_raise(qouts[i], qerrs[i])
                pattern = "permanently stuck"
                self.assertTrue(str(err).find(pattern) > 0)

    def test_normal_flow_with_worker_stats(self):
        """
        Test a very simple 4 trainer case, where elastic_train_step
        also returns a non-None WorkerStats instance.
        """
        run_id = self._generate_run_id()

        nprocs = 4
        qouts = []
        qerrs = []
        prog_rates = [100, 95, 42, None]

        CoordinatorP2P.MONITOR_PROGRESS_FREQ = 1
        original_monitor_progress = CoordinatorP2P.monitor_progress

        def patched_monitor_progress(self, state, worker_stats):
            original_monitor_progress(self, state, worker_stats)

            # Save into state for retrieval in `_get_or_raise` below.
            if hasattr(self, "last_relative_prog_rate"):
                state._test_relative_prog_rate = self.last_relative_prog_rate
            if hasattr(self, "is_worker_straggler"):
                state._test_is_straggler = self.is_worker_straggler

        with patch.object(CoordinatorP2P, "monitor_progress", patched_monitor_progress):
            for i in range(0, nprocs):
                _, qout, qerr = self._spawn(
                    self._train_with_worker_stats,
                    run_id,
                    _train_step,
                    None,  # hooks
                    None,  # state_override
                    prog_rates[i],  # worker_stats_progress_rate
                )
                qouts.append(qout)
                qerrs.append(qerr)

        # Gather all final states, do basic sanity check:
        for i in range(nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            self.assertEqual(5, len(state.nums))

            if i == 3:
                # Rank 3 was hardcoded not to report progress rate
                self.assertFalse(hasattr(state, "_test_relative_prog_rate"))
            else:
                # Other ranks should have expected progress rate
                self.assertTrue(hasattr(state, "_test_relative_prog_rate"))
                self.assertAlmostEqual(
                    state._test_relative_prog_rate,
                    prog_rates[i] / max(pr for pr in prog_rates if pr is not None),
                    delta=1e-5,
                )
            if i == 2:
                # Rank 2 was hardcoded to be a straggler
                self.assertTrue(state._test_is_straggler)

    def test_start_with_min_nodes(self):
        """
        Test elasticity with a max of 4 trainers, but only spawn 2.
        """
        run_id = self._generate_run_id()

        nprocs = 2
        qouts = []
        qerrs = []

        for _ in range(0, nprocs):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None)
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all "trained" values from all trainers
        nums = {}
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            self.assertEqual(10, len(state.nums))
            nums[state.get_worker_rank()] = state.nums
            sums.append(state.total_sum)

        # 2 trainers = 10 samples each, alternating
        # (due to no re-rendezvous, we can assume rank-stability)
        self.assertEqual([11, 13, 15, 17, 19, 21, 23, 25, 27, 29], nums[0])
        self.assertEqual([12, 14, 16, 18, 20, 22, 24, 26, 28, 30], nums[1])
        self.assertEqual([410, 410], sums)

    def test_process_rerendezvous(self):
        """
        Test using a special Coordinator implementation that re-rendezvous
        on every iteration.
        """
        run_id = self._generate_run_id()
        self.min_size = 4  # We expect everyone to join every rendezvous

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs):
            _, qout, qerr = self._spawn(
                self._train_rerendezvous, run_id, _train_step, None
            )
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all "trained" values from all trainers
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            # All 4 trainers should train 5 samples.
            self.assertEqual(5, len(state.nums))
            sums.append(state.total_sum)

        # We re-rendezvous on every iteration, but result should be as normal.
        self.assertEqual([410, 410, 410, 410], sums)

    def test_process_rerendezvous_after_closed(self):
        run_id = self._generate_run_id()

        nprocs = 4
        qouts = []
        qerrs = []

        def train_with_non_aligned_dataset(_, run_id, train_step, hooks):
            state = TestState()
            # generate a dataset that cannot be equally divided, eg: [11:33], there
            # will be 22 elements, cannot be divided by 4 trainers, in this case
            # 2 trainers got last data while the other 2 will hit EOF and exit,
            # the early existed trainer will close rendezvous.
            state.dataset = TestDataset(11, 33)
            return self._train(_, run_id, train_step, hooks, state)

        for _ in range(0, nprocs):
            _, qout, qerr = self._spawn(
                train_with_non_aligned_dataset, run_id, _train_step, None
            )
            qouts.append(qout)
            qerrs.append(qerr)

        # get the samples that each worker processed and assert against input data
        nums = {}
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            self.assertEqual(5, len(state.nums))
            nums[state.get_worker_rank()] = state.nums
            sums.append(state.total_sum)

        self._wait_all_and_clean()
        # created a new trainer, it should exit directly as training complete
        _, qout, qerr = self._spawn(
            train_with_non_aligned_dataset, run_id, _train_step, None
        )
        qouts.append(qout)
        qerrs.append(qerr)
        self.assertEqual(5, len(state.nums))

        # All 4 trainers should train 5 samples without issue
        # (due to no re-rendezvous, we can assume rank-stability)
        self.assertEqual([11, 15, 19, 23, 27], nums[0])
        self.assertEqual([12, 16, 20, 24, 28], nums[1])
        self.assertEqual([13, 17, 21, 25, 29], nums[2])
        self.assertEqual([14, 18, 22, 26, 30], nums[3])
        self.assertEqual([410, 410, 410, 410], sums)  # 410 = 11 + 12 + ... + 30

    def test_process_retryable_exception(self):
        """
        Test 4 trainers, 2 of which throw a retryable exception during
        training and recover.
        """
        run_id = self._generate_run_id()
        self.min_size = 4  # We expect everyone to recover

        def process_retryable_exception():
            # Raise exception only once
            if _train_step.steps == _RUN_HOOKS_AFTER_STEP:
                raise RuntimeError(
                    "train_step throws RuntimeError (retryable exception)"
                )

        hooks = {"process_retryable_exception": process_retryable_exception}

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs - 2):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None)
            qouts.append(qout)
            qerrs.append(qerr)
        for _ in range(nprocs - 2, nprocs):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, hooks)
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all "trained" values from all trainers
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            # All trainers see the expected 20/4=5 samples plus an additional
            # one due to the (retryable excepiton + rollback)
            self.assertEqual(6, len(state.nums))
            sums.append(state.total_sum)

        # We recover from retryable exception - final results same as normal.
        self.assertEqual([410, 410, 410, 410], sums)

    def test_process_non_retryable_exception(self):
        """
        Test 4 trainers, 2 of which throw a non-retryable exception during
        training and terminate early.
        """
        run_id = self._generate_run_id()
        self.min_size = 2  # Only 2 workers expected to recover

        def process_non_retryable_exception():
            # induce a non retryable exception
            baz = bar  # noqa F841 F821

        hooks = {"process_non_retryable_exception": process_non_retryable_exception}

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs - 2):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None)
            qouts.append(qout)
            qerrs.append(qerr)

        for _ in range(nprocs - 2, nprocs):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, hooks)
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all "trained" values from all trainers, and ensure
        # that the bad trainers raise the expected exception.
        sums = []
        for i in range(0, nprocs):
            if i <= 1:
                state = _get_or_raise(qouts[i], qerrs[i])
                sums.append(state.total_sum)
            else:
                with self.assertRaises(NameError):
                    state = _get_or_raise(qouts[i], qerrs[i])

        # Trainers 0 and 1 should process 9 samples because trainers 2 & 3
        # process 3 samples but fail on the 3rd, which causes a rollback.
        # Math:
        #   4 trainers perform two committed train_steps, which adds up to 116.
        #   2 trainers do the remaining steps (19+..+30 = 294)
        #   -> Total 410, i.e. same as normal (i.e this demonstrates recovery)
        self.assertEqual([410, 410], sums)

    def test_process_max_retryable_failures(self):
        """
        Test 4 trainers, 2 of which throw multiple retryable exceptions during
        training and exceed the retry cap.
        """
        run_id = self._generate_run_id()

        def process_retryable_exception():
            # Raise exception repeatedly
            raise RuntimeError("train_step throws RuntimeError (retryable exception)")

        hooks = {"process_retryable_exception": process_retryable_exception}

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs - 2):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None)
            qouts.append(qout)
            qerrs.append(qerr)

        with patch.object(elastic_train_loop, "MAX_FAILURES", 5):
            for _ in range(nprocs - 2, nprocs):
                _, qout, qerr = self._spawn(self._train, run_id, _train_step, hooks)
                qouts.append(qout)
                qerrs.append(qerr)

        # Gather all "trained" values from all trainers, and ensure
        # that the bad trainers raise the expected exception.
        sums = []
        for i in range(0, nprocs):
            if i <= 1:
                state = _get_or_raise(qouts[i], qerrs[i])
                sums.append(state.total_sum)
                # Initially, 4 trainers consume 2 samples each, then the
                # surviving 2 trainers divide the remaining 20-8=12 samples, so
                # the surviving trainers each successfully process 2+6=8 samples.
                # nums keeps track of the samples "seen" so the surviving trainers
                # see an extra 5 samples (one for each retryable exception)
                self.assertEqual(8 + 5, len(state.nums))
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "Exceeded max number of recoverable failures: 5"
                ):
                    state = _get_or_raise(qouts[i], qerrs[i])

        # We completely recover the whole job with 2 out of 4 trainers.
        self.assertEqual([410, 410], sums)

    def test_sync_retryable_exception(self):
        """
        Test 4 trainers, 2 of which throw retryable exceptions during the
        `sync()`` method and recover.
        """
        run_id = self._generate_run_id()

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs - 2):
            _, qout, qerr = self._spawn(
                self._train_rerendezvous, run_id, _train_step, None
            )
            qouts.append(qout)
            qerrs.append(qerr)
        for _ in range(nprocs - 2, nprocs):
            state = TestStateFailOnSync(
                RuntimeError, "sync throws RuntimeError (retryable exception)"
            )
            _, qout, qerr = self._spawn(
                self._train_rerendezvous, run_id, _train_step, None, state
            )
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all the nums from final states, they should match the input
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            self.assertEqual(5, len(state.nums))
            sums.append(state.total_sum)

        # All 4 trainers should train 5 samples because all exceptions
        # are retryable / non-fatal
        self.assertEqual([410, 410, 410, 410], sums)

    def test_checkpoint(self):
        """
        Test with 4 trainers:
            - Save checkpoint every train_step
            - Trainers suicide at 3rd step
            - Restart training (from checkpoint)
        """

        def process_crash():
            log.warning("Suicide, pid:{}".format(os.getpid()))
            os.kill(os.getpid(), signal.SIGKILL)

        hooks = {"process_crash": process_crash}
        run_id = self._generate_run_id()

        nprocs = 4

        # Before training, there is no checkpoint
        checkpoint_manager = FileSystemCheckpointManager(self.test_dir.name)
        self.assertEqual(0, len(checkpoint_manager.list_checkpoints()))

        for _ in range(0, nprocs):
            _, qout, qerr = self._spawn(
                self._train_with_checkpoint, run_id, _train_step, hooks
            )

        # wait all training process complete
        # clean up for next run
        self._wait_all_and_clean()

        # we run 2 steps before suicide, expect two checkpoints be saved
        self.assertEqual(2, len(checkpoint_manager.list_checkpoints()))

        qouts = []
        qerrs = []
        # start next run
        for _ in range(0, nprocs):
            _, qout, qerr = self._spawn(
                self._train_with_checkpoint, run_id, _train_step, None
            )
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all nums and sums from final states, they should match the input
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            # Everyone reads 3 samples after recovering from checkpoint:
            self.assertEqual(3, len(state.nums))
            sums.append(state.total_sum)

        # The job should be completely recovered through checkpoints / crashes:
        self.assertEqual([410, 410, 410, 410], sums)

    def test_process_crash(self):
        """
        Test 4 trainers, 2 of which SIGKILL themselves and terminate.
        """
        run_id = self._generate_run_id()

        def process_crash():
            os.kill(os.getpid(), signal.SIGKILL)

        hooks = {"process_crash": process_crash}

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs - 2):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None)
            qouts.append(qout)
            qerrs.append(qerr)
        for _ in range(nprocs - 2, nprocs):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, hooks)
            qouts.append(qout)
            qerrs.append(qerr)

        sums = []
        for i in range(0, nprocs - 2):
            state = _get_or_raise(qouts[i], qerrs[i])
            sums.append(state.total_sum)

        # 2 surviving workers completely recover and finish the job without loss:
        self.assertEqual([410, 410], sums)

    @unittest.skipIf(
        os.getenv("PYTORCH_TEST_WITH_TSAN", "0") == "1",
        "Skip testing stuck processes with TSAN "
        "because it doesn't like the hung thread.",
    )
    def test_stuck_process(self):
        """
        Test 4 trainers, 2 of which get stuck in an infinite loop.
        """
        run_id = self._generate_run_id()

        def process_stuck():
            # Use infinite loop to simulate a process stucks
            a = 0
            while True:
                a = a + 1

        hooks = {"process_stuck": process_stuck}

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs - 2):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None)
            qouts.append(qout)
            qerrs.append(qerr)

        for _ in range(nprocs - 2, nprocs):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, hooks)
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all the nums from final states, they should match the input except
        # the first two iterations process 2 and 3 consumed data
        sums = []
        for i in range(0, nprocs - 2):
            state = _get_or_raise(qouts[i], qerrs[i])
            sums.append(state.total_sum)

        # Trainers 0 and 1 should process 9 samples because trainers 2 & 3
        # process 3 samples but fail on the 3rd, which causes a rollback, which
        # in turn means that the highest trained sample is '16'.
        self.assertEqual([410, 410], sums)

    def test_rollback_not_supported(self):
        """
        Test 4 trainers, 2 of which throw a retryable exception during
        training and recover. Since the stat's snapshot() method returns None,
        rollback is essentially disabled.
        """
        run_id = self._generate_run_id()

        def process_retryable_exception():
            # Raise exception only once
            if _train_step.steps == _RUN_HOOKS_AFTER_STEP:
                raise RuntimeError(
                    "train_step throws RuntimeError (retryable exception)"
                )

        hooks = {"process_retryable_exception": process_retryable_exception}

        nprocs = 4
        qouts = []
        qerrs = []

        for _ in range(0, nprocs - 2):
            state = TestStateWithRollbackDisabled()
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, None, state)
            qouts.append(qout)
            qerrs.append(qerr)
        for _ in range(nprocs - 2, nprocs):
            state = TestStateWithRollbackDisabled()
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, hooks, state)
            qouts.append(qout)
            qerrs.append(qerr)

        # Gather all "trained" values from all trainers
        # half of the trainers throw an exception on the 3rd train_step
        # with no rollback, effectively making the 3rd example on those workers
        # "unsuccessful". Hence ``nums`` on those workers is one less than
        # ``nums`` on the always successful workers.
        sums = []
        for i in range(0, nprocs):
            state = _get_or_raise(qouts[i], qerrs[i])
            self.assertEqual(5, len(state.nums))
            sums.append(state.total_sum)

        self.assertEqual([410, 410, 410, 410], sums)

    def test_new_processes_join(self):
        """
        Test that 2 new processes can join an existing elastic process group
        and do work.
        """

        # Kick off 2 workers to do some work, and after 3 iters have them
        # block until we can launch 2 more workers. Once all 4 workers are
        # ready, continue training.

        run_id = self._generate_run_id()

        # This semaphore blocks the master process from spawning the last
        # 2 workers and is only released when the first 2 workers have
        # processed 3 samples each.
        sem_spawn_new_workers = multiprocessing.Semaphore(0)
        # This semaphore is used to block the first 2 workers until
        # the 2 additional workers are ready to rendezvous.
        sem_resume_working = multiprocessing.Semaphore(0)

        def wait_for_new_processes():
            # Hooks run on & after the 3rd iteration, but do this only once.
            if _train_step.steps == _RUN_HOOKS_AFTER_STEP:
                # Signal to master to spawn more workers
                sem_spawn_new_workers.release()
                # Wait for those new workers to be ready
                sem_resume_working.acquire()

                # The signal that new workers are waiting is "eventually propagated",
                # without extra synchronization, i.e. there's a bit of a race here.
                time.sleep(5)

        hooks = {"wait_for_new_processes": wait_for_new_processes}

        qouts = []
        qerrs = []

        state = TestState(RadixTestDataset(max_iter=6))

        for _ in range(0, 2):
            _, qout, qerr = self._spawn(self._train, run_id, _train_step, hooks, state)
            qouts.append(qout)
            qerrs.append(qerr)

        # Wait for both workers to notify us
        sem_spawn_new_workers.acquire()
        sem_spawn_new_workers.acquire()

        orig_rendezvous_barrier = CoordinatorP2P.rendezvous_barrier

        def patched_rendezvous_barrier(self):
            # Notify 1 waiting worker (this will happen twice total)
            sem_resume_working.release()
            return orig_rendezvous_barrier(self)

        with patch.object(
            CoordinatorP2P, "rendezvous_barrier", patched_rendezvous_barrier
        ):
            for _ in range(0, 2):
                _, qout, qerr = self._spawn(
                    self._train, run_id, _train_step, None, state
                )
                qouts.append(qout)
                qerrs.append(qerr)

        sums = []
        for i in range(0, 4):
            state = _get_or_raise(qouts[i], qerrs[i])
            sums.append(state.total_sum)

        # The first three iterations are executed by two workers
        early_iter = RadixTestDataset.get_expected_sum(3, [0, 1])
        # The rest of the workload is distributed to four workers
        late_iter = RadixTestDataset.get_expected_sum(7, [0, 1, 2, 3], start_iter=4)
        for sum in sums:
            self.assertEqual(sum, early_iter + late_iter)


if __name__ == "__main__":
    unittest.main()
