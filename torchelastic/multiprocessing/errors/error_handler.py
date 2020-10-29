#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

import json
import os
import signal
import time
import traceback
from typing import Optional


class ProcessException(Exception):
    def __init__(
        self,
        msg: str,
        pid: int,
        timestamp: int = 0,
        exit_code: Optional[int] = None,
        signal_name: Optional[str] = None,
    ):
        super().__init__(msg)
        self.pid = pid
        self.timestamp = timestamp
        self.exit_code = exit_code
        self.signal_name = signal_name


class ErrorHandler:
    def __init__(self, log_dir: str = "/tmp/torchelastic/logs"):
        self._processes = []
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def configure(self, root_process: bool = False) -> None:
        """
        Configures necessary behavior on the process. This is a separate
        from the constructor since child and parent processes
        can call this method with different arguments
        """
        pass

    def try_raise_exception(self, child_pid: int, exit_code: int = 1) -> None:
        """
        Tries to retrieve the exception using ``child_pid``. If the child process
        recorded exception using ``torchelastic.multiprocessing.errors`` module,
        the raised exception will contain the root cause message.
        """
        child_error_file = self._get_error_file_path(child_pid)
        signal_name = self._try_get_signal_name(exit_code)
        assert child_error_file is not None
        if not os.path.exists(child_error_file):
            error_timestamp = int(time.time() * 1000)
            message = f"Process {child_pid} terminated with exit code {exit_code}, signal name: {signal_name}"
        else:
            error_timestamp = int(os.path.getmtime(child_error_file) * 1000)
            child_message = self._get_error_message(child_error_file)
            message = f"Process {child_pid} terminated with exit code {exit_code}, signal name: {signal_name}, message: {child_message}"
        raise ProcessException(
            message,
            pid=child_pid,
            timestamp=error_timestamp,
            exit_code=exit_code,
        )

    def record_exception(self, e: BaseException, root_process: bool = False) -> None:
        """
        Records the exception that can be retrieved later by the parent process
        """
        error_file = self._get_error_file_path(os.getpid())
        message = traceback.format_exc()
        data = {"message": message}
        assert error_file is not None
        with open(error_file, "w") as f:
            json.dump(data, f)

    def _try_get_signal_name(self, exit_code: int) -> Optional[str]:
        if exit_code < 0:
            return signal.Signals(-exit_code).name
        else:
            return None

    def _get_error_message(self, error_file) -> Optional[str]:
        if not os.path.exists(error_file):
            return None
        with open(error_file, "r") as f:
            data = json.load(f)
            return data["message"]

    def _get_error_file_path(
        self, pid: int, root_process: bool = False
    ) -> Optional[str]:
        process_dir = f"{self.log_dir}/{pid}"
        os.makedirs(process_dir, exist_ok=True)
        return f"{process_dir}/error.log"
