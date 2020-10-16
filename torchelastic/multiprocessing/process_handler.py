#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import signal
from typing import Optional

from torchelastic.multiprocessing.base_process_handler import (
    BaseProcessHandler,
    ProcessExitedException,
)


class ProcessHandler(BaseProcessHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _wait_or_raise(self, timeout: Optional[float] = None):
        return_code = super().wait(timeout)
        if return_code != 0:
            signal_name = None
            if return_code < 0:
                signal_name = signal.Signals(-return_code).name
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
        Waits for child process to terminate. Raises exception if
        process exited with non-zero code.
        """
        return self._wait_or_raise(timeout)
