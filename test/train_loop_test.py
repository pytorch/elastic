#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import typing
import unittest
from dataclasses import dataclass

from torchelastic.train_loop import to_generator


class TestTrainLoop(unittest.TestCase):
    @dataclass
    class MockState:
        data: typing.Generator[int, None, None]
        sum: int = 0

    @staticmethod
    def _mock_train_step(mock_state):
        mock_state.sum += next(mock_state.data)

    @staticmethod
    def _mock_train_step_stop_iteration(mock_state):
        raise StopIteration

    def test_to_generator(self):
        r = range(0, 10)
        mock_state = self.MockState(iter(r))

        for _ in to_generator(self._mock_train_step)(mock_state):
            pass

        self.assertEqual(sum(r), mock_state.sum)

    def test_to_generator_throws_stop_iteration(self):

        mock_state = self.MockState(iter(range(0, 10)))

        num_iter = 0
        for _ in to_generator(self._mock_train_step_stop_iteration)(mock_state):
            num_iter += 1

        # _mock_train_step_stop_iteration throws StopIteration
        # as soon as it invokes so we expect the generator to stop immediately
        self.assertEqual(0, num_iter)
