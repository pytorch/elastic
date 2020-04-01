#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
import warnings
from collections import namedtuple
from functools import wraps


MetricData = namedtuple("MetricData", ["timestamp", "group_name", "name", "value"])


class MetricHandler(abc.ABC):
    @abc.abstractmethod
    def emit(self, metric_data: MetricData):
        pass


class ConsoleMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData):
        print(
            "[{}][{}]: {}={}".format(
                metric_data.timestamp,
                metric_data.group_name,
                metric_data.name,
                metric_data.value,
            )
        )


class NullMetricHandler(MetricHandler):
    def emit(self, metric_data: MetricData):
        pass


class MetricStream:
    def __init__(self, group_name: str, handler: MetricHandler):
        self.group_name = group_name
        self.handler = handler

    def add_value(self, metric_name: str, metric_value: int):
        self.handler.emit(
            MetricData(time.time(), self.group_name, metric_name, metric_value)
        )


_metrics_map = {}
_default_metrics_handler = NullMetricHandler()


def configure(handler: MetricHandler, group: str = None):
    if group is None:
        global _default_metrics_handler
        _default_metrics_handler = handler
    else:
        _metrics_map[group] = handler


def getStream(group: str):
    if group in _metrics_map:
        handler = _metrics_map[group]
    else:
        handler = _default_metrics_handler
    return MetricStream(group, handler)


def _get_metric_name(fn):
    qualname = fn.__qualname__
    split = qualname.split(".")
    if len(split) == 1:
        module = fn.__module__
        if module:
            return module.split(".")[-1] + "." + split[0]
        else:
            return split[0]
    else:
        return qualname


def prof(fn=None, group: str = "torchelastic"):
    r"""
    @profile decorator publishes duration.ms, count, success, failure
    metrics for the function that it decorates. The metric name defaults
    to the qualified name (``class_name.def_name``) of the function.
    If the function does not belong to a class, it uses the leaf module name
    instead.

    Usage

    ::

     @metrics.prof
     def x():
         pass

     @metrics.prof(group="agent")
     def y():
         pass
    """

    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            key = _get_metric_name(f)
            try:
                start = time.time()
                result = f(*args, **kwargs)
                put_metric(f"{key}.success", 1, group)
            except Exception:
                put_metric(f"{key}.failure", 1, group)
                raise
            finally:
                put_metric(f"{key}.duration.ms", get_elapsed_time_ms(start), group)
            return result

        return wrapper

    if fn:
        return wrap(fn)
    else:
        return wrap


def profile(group=None):
    """
    @profile decorator adds latency and success/failure metrics to any given function.

    Usage

    ::

     @metrics.profile("my_metric_group")
     def some_function(<arguments>):
    """
    warnings.warn("Deprecated, use @prof instead", DeprecationWarning)

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


def put_metric(metric_name: str, metric_value: int, metric_group: str = "torchelastic"):
    """
    Publishes a metric data point.

    Usage

    ::

     put_metric("metric_name", 1)
     put_metric("metric_name", 1, "metric_group_name")
    """

    getStream(metric_group).add_value(metric_name, metric_value)


def publish_metric(metric_group: str, metric_name: str, metric_value: int):
    warnings.warn(
        "Deprecated, use put_metric(metric_group)(metric_name, metric_value) instead"
    )
    metric_stream = getStream(metric_group)
    metric_stream.add_value(metric_name, metric_value)


def get_elapsed_time_ms(start_time_in_seconds: float):
    """
    Returns the elapsed time in millis from the given start time.
    """
    end_time = time.time()
    return int((end_time - start_time_in_seconds) * 1000)
