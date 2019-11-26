#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
from collections import namedtuple
from functools import wraps


MetricData = namedtuple("MetricData", ["timestamp", "group_name", "name", "value"])


class MetricHandler(abc.ABC):
    @abc.abstractmethod
    def emit(self, metric_data):
        pass


class ConsoleMetricHandler(MetricHandler):
    def emit(self, metric_data):
        print(
            "[{}][{}]: {}={}".format(
                metric_data.timestamp,
                metric_data.group_name,
                metric_data.name,
                metric_data.value,
            )
        )


class NullMetricHandler(MetricHandler):
    def emit(self, metric_data):
        pass


class MetricStream:
    def __init__(self, group_name, handler):
        self.group_name = group_name
        self.handler = handler

    def add_value(self, metric_name, metric_value):
        self.handler.emit(
            MetricData(time.time(), self.group_name, metric_name, metric_value)
        )


_metrics_map = {}
_default_metrics_handler = NullMetricHandler()


def configure(handler, group=None):
    if group is None:
        global _default_metrics_handler
        _default_metrics_handler = handler
    else:
        _metrics_map[group] = handler


def getStream(group):
    if group in _metrics_map:
        handler = _metrics_map[group]
    else:
        handler = _default_metrics_handler
    return MetricStream(group, handler)


def profile(group=None):
    """
    @profile decorator adds latency and success/failure metrics to any given function.


    Typical usage:
        @metrics.profile("my_metric_group")
        def some_function(<arguments>):
    """

    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                publish_metric(group, "{}.success".format(func.__name__), 1)
            except Exception:
                publish_metric(group, "{}.failure".format(func.__name__), 1)
                raise
            finally:
                publish_metric(
                    group,
                    "{}.duration.ms".format(func.__name__),
                    get_elapsed_time_ms(start_time),
                )
            return result

        return wrapper

    return wrap


def publish_metric(metric_group, metric_name, metric_value):
    metric_stream = getStream(metric_group)
    metric_stream.add_value(metric_name, metric_value)


def get_elapsed_time_ms(start_time_in_seconds):
    """
    Returns the elapsed time in millis from the given start time.
    """
    end_time = time.time()
    return int((end_time - start_time_in_seconds) * 1000)
