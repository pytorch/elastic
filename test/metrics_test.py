#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torchelastic.metrics as metrics


class MockMetricHandler(metrics.MetricHandler):
    def __init__(self, metric_map):
        self.metric_map = metric_map

    def emit(self, metric_data):
        self.metric_map[
            f"{metric_data.group_name}.{metric_data.name}"
        ] = metric_data.value


metric_map = {}
metrics.configure(MockMetricHandler(metric_map), "torchelastic")


class MetricsTest(unittest.TestCase):
    def test_metrics_decorator(self):
        @metrics.profile("torchelastic")
        def func(a, b):
            return a + b

        result = func(3, 2)
        self.assertTrue(5, result)
        self.assertTrue("torchelastic.func.duration.ms" in metric_map)
        self.assertEqual(1, metric_map["torchelastic.func.success"])

        # Calls the decorator as a function
        def abc(a, b):
            return a + b, a * b

        result = metrics.profile("torchelastic")(abc)(3, 2)
        self.assertTrue(5, result[0])
        self.assertTrue(6, result[1])
        self.assertTrue("torchelastic.abc.duration.ms" in metric_map)
        self.assertEqual(1, metric_map["torchelastic.abc.success"])

        @metrics.profile("torchelastic")
        def xyz():
            raise Exception("Thrown from xyz")

        try:
            xyz()
        except Exception:
            # Expected
            pass

        self.assertTrue("torchelastic.xyz.duration.ms" in metric_map)
        self.assertEqual(1, metric_map["torchelastic.xyz.failure"])

    def test_default_metrics(self):
        # default is the null handler, simply check that
        # adding to a un-configured metric group does not throw
        metrics_stream = metrics.getStream("default")
        metrics_stream.add_value("torchelastic_test.counter", 2)
        # nothing to assert, counter goes to a black hole sink

    def test_override_default_metric_handler(self):
        metrics.configure(MockMetricHandler(metric_map))
        root_stream = metrics.getStream("default")
        root_stream.add_value("test.counter", 3)
        self.assertEqual(3, metric_map["default.test.counter"])
