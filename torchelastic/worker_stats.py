#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc


class WorkerStats(abc.ABC):
    """
    Data-oriented class used to return stats produced by application's `train_step` and
    made observable for the elastic trainer loop.
    """

    @abc.abstractmethod
    def get_progress_rate(self):
        """
        Returns a numerical representation of rate at which the worker is doing work.
        Bigger numerical value is interpreted as is working faster.
        """

        pass


class SimpleWorkerStats(WorkerStats):
    """
    Simple implementation of WorkerStats that takes as
    constructor argument a numerical progress_rate and
    returns it when the get_progress_rate() method is called.
    """

    def __init__(self, progress_rate: float):
        self._progress_rate = progress_rate

    def get_progress_rate(self):
        return self._progress_rate
