#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from abc import ABC, abstractmethod
from typing import Optional


# TODO(aivanou): move this to subprocess_handler.py and add Subprocess prefix
class ProcessException(Exception):
    def __init__(
        self, msg: str, pid: int, stdout=None, stderr=None, timestamp: int = 0
    ):
        super().__init__(msg)
        self._pid = pid
        self.stdout = stdout
        self.stderr = stderr
        self.timestamp = timestamp


class ProcessRaisedException(ProcessException):
    def __init__(
        self, msg: str, pid: int, stdout=None, stderr=None, timestamp: int = 0
    ):
        super().__init__(msg, pid, stdout, stderr, timestamp)


class ProcessExitedException(ProcessException):
    def __init__(
        self,
        msg: str,
        pid: int,
        exit_code: int,
        signal_name: Optional[str] = None,
        stdout=None,
        stderr=None,
        timestamp: int = 0,
    ):
        super().__init__(msg, pid, stdout, stderr, timestamp)
        self.exit_code = exit_code
        self.signal_name = signal_name


class BaseProcessHandler(ABC, subprocess.Popen):
    @abstractmethod
    def wait_or_raise(self, timeout: Optional[float] = None):
        """
        Waits for child process to terminate. Raises exception if
        process exited with non-zero code.
        """
        raise NotImplementedError

    def is_alive(self) -> bool:
        """
        Returns True if process is running, otherwise returns False.
        """
        return self.poll() is None
