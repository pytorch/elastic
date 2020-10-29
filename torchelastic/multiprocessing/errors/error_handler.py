#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

import json
import logging
import os
import signal
import time
import traceback
from typing import Optional


logger: logging.Logger = logging.getLogger(__name__)


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
    def __init__(self, error_dir: str = "/tmp/torchelastic/logs"):
        self._processes = []
        self.error_dir = error_dir

    def configure(self) -> None:
        """
        Configures necessary behavior on the process. This is a separate
        from the constructor since child and parent processes
        can call this method with different arguments
        """
        if not os.path.exists(self.error_dir):
            self._try_create_dirs(self.error_dir)

    def _try_create_dirs(self, dir: str):
        try:
            os.makedirs(self.error_dir, exist_ok=True)
        except Exception as e:
            logger.info(f"Wasn't able to create {self.error_dir}. Failed with: {e}")

    def try_raise_exception(self, child_pid: int, exit_code: int = 1) -> None:
        """
        Tries to retrieve the exception using ``child_pid``. If the child process
        recorded exception using ``torchelastic.multiprocessing.errors`` module,
        the raised exception will contain the root cause message.
        """
        child_error_file = self._get_error_reply_file(child_pid)
        signal_name = self._try_get_signal_name(exit_code)
        message = f"Process {child_pid} terminated with exit code {exit_code}"
        if signal_name:
            message += f", signal name: {signal_name}"
        if not os.path.exists(child_error_file):
            logger.info(
                f"Reply file: {child_error_file} does not exist, raising generic message"
            )
            error_timestamp = int(time.time() * 1000)
        else:
            logger.info(
                f"Reply file: {child_error_file} exists, raising specific message"
            )
            error_timestamp = int(os.path.getmtime(child_error_file) * 1000)
            child_message = self._get_error_message(child_error_file)
            message += f"worker message: {child_message}"
        raise ProcessException(
            message,
            pid=child_pid,
            timestamp=error_timestamp,
            exit_code=exit_code,
        )

    def _get_error_reply_file(self, pid: int):
        return f"{self.error_dir}/error_{pid}.log"

    def record_exception(self, e: BaseException) -> None:
        """
        Records the exception that can be retrieved later by the parent process
        """
        error_file = self._get_error_reply_file(os.getpid())
        message = traceback.format_exc()
        data = {"message": message}
        assert error_file is not None
        with open(error_file, "w") as f:
            json.dump(data, f)

    def get_error_dir(self) -> str:
        """
        Returns the error dir that should be the same across parent and child processes.
        """
        return self.error_dir

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
