#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

from functools import wraps

from torchelastic.multiprocessing.errors.handlers import get_error_handler


_process_error_handler = get_error_handler()


def try_raise_exception(error_process_pid: int, exit_code: int = 0) -> None:
    """
    Tries to retrieve the exception that was recorded by the ``error_process_pid``
    and raises it as ``ProcessException``. exit_code can be provided that will be
    used to build a message as well as to retrieve signal name.
    """
    return _process_error_handler.try_raise_exception(error_process_pid, exit_code)


def get_error_dir() -> str:
    """
    Tries to retrieve the exception that was recorded by the ``error_process_pid``
    and raises it as ``ProcessException``. exit_code can be provided that will be
    used to build a message as well as to retrieve signal name.
    """
    return _process_error_handler.get_error_dir()


def record(fn=None):
    """
    Decorator function that is invoked on the starting process function.
    The decorator will invoke signal registerring mechanism that will allow
    to propagate any signal termination related errors to the parent process.

    Note: the decorator should be invoked a single time per processor, and
    it is best to set it on top of the main function.

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
