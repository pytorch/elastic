#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import signal
import subprocess
from typing import Optional

import torchelastic.multiprocessing.error_reporter as error_reporter


class ProcessException(Exception):
    def __init__(self, msg: str, pid: int, stdout=None, stderr=None):
        super().__init__(msg)
        self._pid = pid
        self.stdout = stdout
        self.stderr = stderr


class ProcessRaisedException(ProcessException):
    def __init__(self, msg: str, pid: int, stdout=None, stderr=None):
        super().__init__(msg, pid, stdout, stderr)


class ProcessExitedException(ProcessException):
    def __init__(
        self,
        msg: str,
        pid: int,
        exit_code: int,
        signal_name: Optional[str] = None,
        stdout=None,
        stderr=None,
    ):
        super().__init__(msg, pid, stdout, stderr)
        self.exit_code = exit_code
        self.signal_name = signal_name


class ResponsivePopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_signal_name_from_return_code(self, return_code: int) -> Optional[str]:
        if return_code < 0:
            return signal.Signals(-return_code).name
        return None

    def _wait_or_raise(self, timeout: Optional[float] = None):
        return_code = super().wait(timeout)
        process_error = error_reporter.get_error(self.pid)
        if process_error:
            if process_error.error_type == error_reporter.ErrorType.MANAGED:
                raise ProcessRaisedException(
                    process_error.message, self.pid, self.stdout, self.stderr
                )
            else:
                signal_name = self._get_signal_name_from_return_code(return_code)
                raise ProcessExitedException(
                    process_error.message,
                    self.pid,
                    return_code,
                    signal_name,
                    self.stdout,
                    self.stderr,
                )
        if return_code != 0:
            signal_name = self._get_signal_name_from_return_code(return_code)
            raise ProcessExitedException(
                f"Process {self.pid} terminated with code {return_code}",
                self.pid,
                return_code,
                signal_name,
                self.stdout,
                self.stderr,
            )
        return return_code

    def wait_or_raise(self, timeout: Optional[float] = None):
        """
        Wait for child process to terminate. After wait is finished, the
        error reporter will check whether there were any exceptions.
        """
        try:
            return self._wait_or_raise(timeout)
        finally:
            error_reporter.cleanup()

    def is_alive(self) -> bool:
        return self.poll() is None
