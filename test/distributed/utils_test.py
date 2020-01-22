#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs

import multiprocessing as mp
import unittest
from unittest.mock import patch

import torchelastic.distributed as edist


def _return_false():
    return False


def _return_true():
    return True


def _return_one():
    return 1


def _get_rank(ignored):
    """
    wrapper around torchelastic.distributed.get_rank()
    take the element in the input argument as parameter
    since multiprocessing.Pool.map requires the function to
    """
    return edist.get_rank()


class TestUtils(unittest.TestCase):
    @patch("torch.distributed.is_available", _return_true)
    @patch("torch.distributed.is_initialized", _return_false)
    def test_get_rank_no_process_group_initialized(self):
        # always return rank 0 when process group is not initialized
        num_procs = 4
        with mp.Pool(num_procs) as p:
            ret = p.map(_get_rank, range(0, num_procs))
            for rank in ret:
                self.assertEqual(0, rank)

    @patch("torch.distributed.is_available", _return_false)
    @patch("torch.distributed.is_initialized", _return_true)
    def test_get_rank_no_dist_available(self):
        # always return rank 0 when distributed torch is not available
        num_procs = 4
        with mp.Pool(num_procs) as p:
            ret = p.map(_get_rank, range(0, num_procs))
            for rank in ret:
                self.assertEqual(0, rank)

    @patch("torch.distributed.is_available", _return_true)
    @patch("torch.distributed.is_initialized", _return_true)
    @patch("torch.distributed.get_rank", _return_one)
    def test_get_rank(self):
        world_size = 4
        with mp.Pool(world_size) as p:
            ret = p.map(_get_rank, range(0, world_size))

        # since we mocked a return value of 1
        # from torch.distributed.get_rank()
        # we expect that the sum of ranks == world_size
        self.assertEqual(world_size, sum(ret))
