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
import shutil
import signal
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Optional


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ProcessFailure:
    """
    Represents the failed process result. When the worker process fails,
    it may record failure root cause into the file.
    """

    error_file: Optional[str]
    timestamp: int  # seconds
    pid: int
    rank: int
    exit_code: int

    def get_signal_name(self) -> Optional[str]:
        if self.exit_code < 0:
            return signal.Signals(-self.exit_code).name
        else:
            return None


class ErrorHandler:
    def __init__(self):
        self.error_file = ""
        self.temp_dir = ""

    def configure(self) -> None:
        """
        Configures necessary behavior on the process. This is a separate
        from the constructor since child and parent processes
        can call this method with different arguments
        """
        if "TORCHELASTIC_ERROR_FILE" in os.environ:
            self.error_file = os.environ["TORCHELASTIC_ERROR_FILE"]
        elif not self.error_file:
            self.temp_dir = tempfile.mkdtemp()
            self.error_file = os.path.join(self.temp_dir, "error.log")
        self._try_create_dirs(os.path.dirname(self.error_file))

    def _try_create_dirs(self, dir: str):
        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            logger.info(f"Wasn't able to create {dir}. Failed with: {e}")

    def get_failed_result(
        self,
        child_rank: int,
        child_pid: int,
        exit_code: int = 1,
        run_id: int = 0,
    ) -> Optional[ProcessFailure]:
        """
        Returns failed result. Tries to retrieve the child error file,
        if no file exists on the path: ``os.path.join(error_dir, rank, error.log)``,
        the error_file will have ``None`` value, otherwise it will contain the path.
        """
        child_error_file = self.get_error_file(child_rank, run_id)
        if child_error_file and os.path.exists(child_error_file):
            error_file = child_error_file
            timestamp = int(os.path.getmtime(child_error_file) * 1000)
        else:
            error_file = None
            timestamp = int(time.time() * 1000)
        return ProcessFailure(
            error_file=error_file,
            timestamp=timestamp,
            pid=child_pid,
            rank=child_rank,
            exit_code=exit_code,
        )

    def get_error_file(self, rank: int, run_id: int = 0) -> str:
        """
        Returns the error dir that should be the same across parent and child processes.
        """
        if not self.error_file:
            return ""
        error_dir = os.path.dirname(self.error_file)
        return os.path.join(error_dir, str(rank), f"error.log_{run_id}")

    def process_failure(self, failure: ProcessFailure) -> None:
        """
        Tries to retrieve the error from the file in ``ProcessFailure.error_file``
        and copy-pastes the content to the parent error file.
        """
        child_error_file = failure.error_file
        if (
            child_error_file is None
            or not self.error_file
            or not os.path.exists(child_error_file)
        ):
            logger.warning(
                f"Worker {failure.rank} exited with exit_code: {failure.exit_code}, but no {child_error_file} found"
            )
        elif self.error_file and os.path.exists(self.error_file):
            logger.warning(f"Error file {self.error_file} already exists, skipping")
        else:
            self._process_failure(child_error_file)

    def _process_failure(self, child_error_file: str):
        logger.info(
            f"Copying worker error file {child_error_file} to {self.error_file}"
        )
        with open(child_error_file, "r") as f:
            data = json.load(f)
        with open(self.error_file, "w") as f:
            json.dump(data, f)

    def _get_failure_message(self, failure: ProcessFailure) -> str:
        msg = f"Worker {failure.rank} failed with exit_code: {failure.exit_code} "
        if failure.get_signal_name():
            msg += f";signal name: {failure.get_signal_name()}"
        return msg

    def record_exception(self, e: BaseException) -> None:
        """
        Records the exception that can be retrieved later by the parent process
        """
        if not self.error_file:
            logger.warning(f"Error file not set for process {os.getpid()}")
            return
        message = traceback.format_exc()
        data = {"message": message}
        with open(self.error_file, "w") as f:
            json.dump(data, f)

    def cleanup(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            logger.info(f"Cleaning up resources at: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
