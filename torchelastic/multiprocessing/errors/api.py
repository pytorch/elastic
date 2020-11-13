#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

from functools import wraps
from typing import Any, Callable, Optional, Tuple

from torchelastic.multiprocessing.errors.error_handler import ProcessFailure
from torchelastic.multiprocessing.errors.handlers import get_error_handler


_process_error_handler = get_error_handler()


def exec_fn(f: Callable, args: Tuple) -> Any:
    """
    Executes provided function with configured error handler. If any exception
    occurs, it will be first processed by the error handler before raising it.
    """
    try:
        _process_error_handler.configure()
        result = f(*args)
        return result
    except Exception as e:
        _process_error_handler.record_exception(e)
        raise


def process_failure(failure: ProcessFailure) -> None:
    """
    Tries to retrieve the error from the file in ``ProcessFailure.error_file``
    and copy-pastes the content to the parent error file. Raises the ``Exception``
    to indicate to the upper stack that the error occurred on child process.
    If no child_error found, the warning message will be printed to the logger and
    no error file will be recorded.
    """
    _process_error_handler.process_failure(failure)


def get_failure_message(failure: ProcessFailure) -> str:
    """
    Retrieves and pretty prints the failure message.
    """
    return _process_error_handler._get_failure_message(failure)


def cleanup() -> None:
    """
    Cleanup resources that may be created by the error handler
    """
    _process_error_handler.cleanup()


def get_failed_result(
    child_rank: int, child_pid: int, exit_code: int = 1, run_id: int = 0
) -> Optional[ProcessFailure]:
    """
    Returns error file
    """
    return _process_error_handler.get_failed_result(
        child_rank, child_pid, exit_code, run_id
    )


def get_error_file(rank: int, run_id: int = 0) -> str:
    """
    Returns error file based on the rank in format:
    ``os.path.join(base_dir, rank, error.log_{run_id})``
    """
    return _process_error_handler.get_error_file(rank, run_id)


def record(fn=None):
    """
    Decorator function that is invoked on the starting process function.
    The decorator will invoke signal registerring mechanism that will allow
    to propagate any signal termination related errors to the parent process.

    Note: the decorator should be invoked a single time per processor, and
    it is best to set it on top of the main function. The decorator invokes
    ``configure`` method to properly configure error handler and
    ``record_exception`` method to record exception. If the error_file
    already exists, the exception will not be recorded.

    Example

    ::

        # child
        from torchelastic.multiprocessing.errors import record

        @record
        def main():
            ...

        if __name__=="__main__":
            main()

        # parent
        from torchelastic.multiprocessing.errors import record

        @record()
        def main():
            # invoke child main as subprocess
            ...

        if __name__=="__main__":
            main()

    """

    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                _process_error_handler.configure()
                result = f(*args, **kwargs)
            except Exception as e:
                _process_error_handler.record_exception(e)
                raise
            return result

        return wrapper

    if fn:
        return wrap(fn)
    else:
        return wrap
