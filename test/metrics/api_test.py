#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs
import abc
import unittest

from torchelastic.metrics.api import (
    MetricData,
    MetricHandler,
    _get_metric_name,
    configure,
    prof,
)


def foo_1():
    pass


class TestMetricsHandler(MetricHandler):
    def __init__(self):
        self.metric_data = {}

    def emit(self, metric_data: MetricData):
        self.metric_data[metric_data.name] = metric_data


class Parent(abc.ABC):
    @abc.abstractmethod
    def func(self):
        raise NotImplementedError()

    def base_func(self):
        self.func()


class Child(Parent):
    # need to decorate the implementation not the abstract method!
    @prof
    def func(self):
        pass


class MetricsApiTest(unittest.TestCase):
    def foo_2(self):
        pass

    @prof
    def bar(self):
        pass

    @prof
    def throw(self):
        raise RuntimeError()

    @prof(group="torchelastic")
    def bar2(self):
        pass

    def test_get_metric_name(self):
        self.assertEqual("api_test.foo_1", _get_metric_name(foo_1))
        self.assertEqual("MetricsApiTest.foo_2", _get_metric_name(self.foo_2))

    def test_profile(self):
        handler = TestMetricsHandler()
        configure(handler)

        self.bar()

        self.assertEqual(1, handler.metric_data["MetricsApiTest.bar.count"].value)
        self.assertEqual(1, handler.metric_data["MetricsApiTest.bar.success"].value)
        self.assertNotIn("MetricsApiTest.bar.failure", handler.metric_data)
        self.assertIn("MetricsApiTest.bar.duration.ms", handler.metric_data)

        with self.assertRaises(RuntimeError):
            self.throw()

        self.assertEqual(1, handler.metric_data["MetricsApiTest.throw.count"].value)
        self.assertEqual(1, handler.metric_data["MetricsApiTest.throw.failure"].value)
        self.assertNotIn("MetricsApiTest.bar_raise.success", handler.metric_data)
        self.assertIn("MetricsApiTest.throw.duration.ms", handler.metric_data)

        self.bar2()
        self.assertEqual(
            "torchelastic", handler.metric_data["MetricsApiTest.bar2.count"].group_name
        )

    def test_inheritance(self):
        handler = TestMetricsHandler()
        configure(handler)

        c = Child()
        c.base_func()

        self.assertEqual(1, handler.metric_data["Child.func.count"].value)
        self.assertEqual(1, handler.metric_data["Child.func.success"].value)
        self.assertIn("Child.func.duration.ms", handler.metric_data)
