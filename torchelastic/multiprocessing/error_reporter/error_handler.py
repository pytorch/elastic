#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module


from abc import ABC, abstractmethod
from typing import Optional


# Error reported uses files to communicate between processes.
# The name of the environment variable that contains path to the file
# with error message and stacktrace raised by child process.
# Each process has a unique error file.
ERROR_FILE_ENV: str = "ERROR_FILE_ENV_VAR"


class ErrorHandler(ABC):
    @abstractmethod
    def configure(self):
        raise NotImplementedError

    @abstractmethod
    def construct_error_message(self, child_pid: int) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def record_exception(self, e: Exception) -> None:
        raise NotImplementedError


class LocalErrorHandler(ErrorHandler):
    """
    Since we use stderr to get the errors for oss, the local error handler does
    do anything.
    """

    def configure(self):
        pass

    def construct_error_message(self, child_pid: int) -> Optional[str]:
        pass

    def record_exception(self, e: Exception) -> None:
        pass
