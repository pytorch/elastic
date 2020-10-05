#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

import logging
import os
from typing import Any, Callable, Optional, Tuple

from torchelastic.multiprocessing.error_reporter.handlers import get_error_handler


log: logging.Logger = logging.getLogger(__name__)


def exec_fn(user_funct: Callable, args: Tuple = ()) -> Any:
    log.warning("This is an experimental API and is going to be removed in future")
    error_handler = get_error_handler()
    error_handler.configure()
    return user_funct(*args)


def get_error(error_process_pid: int) -> Optional[str]:
    """
    Retrieves an error(if any) that occurred on the child process.
    If no exception occurred, the function will return None
    """
    log.warning("This is an experimental API and is going to be removed in future")
    error_handler = get_error_handler()
    return error_handler.construct_error_message(error_process_pid)


def _configure_process_handler() -> None:
    log.warning("This is an experimental API and is going to be removed in future")
    error_handler = get_error_handler()
    error_handler.configure()


def record(func):
    """
    Decorator function that is invoked on the starting process function.
    The decorator will invoke signal registerring mechanism that will allow
    to propagate any signal termination related errors to the parent process.

    Example

    ::

        from torchelastic.multiprocessing.error_reporter import record

        @record
        def main():
            ...

        if __name__=="__main__":
            main()

    """

    log.warning("This is an experimental API and is going to be removed in future")

    def wrap():
        try:
            _configure_process_handler()
            func()
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.record_exception(e)
            raise

    return wrap
