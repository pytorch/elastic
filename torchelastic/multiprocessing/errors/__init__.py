#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Library is used to write and retrieve errors between processes. It contains
methods to record errors on child processes and retrieve errors on the parent processes.

Usage

::

    # parent process

    from torchelastic.multiprocessing.errors import get_error

    child_proc = launch_proc()
    error = gt_error(child_proc.pid)

    # child process

    from torchelastic.multiprocessing.errors import record

    @record
    def main():
        raise RuntimeError("test error")

"""


from torchelastic.multiprocessing.errors.error_handler import (  # noqa F401
    ProcessFailure,
)

from .api import (  # noqa F401
    cleanup,
    exec_fn,
    get_error_file,
    get_failed_result,
    get_failure_message,
    process_failure,
    record,
)
