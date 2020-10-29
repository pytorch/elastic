#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import signal
import subprocess
from typing import Optional

from torchelastic.multiprocessing.errors import try_raise_exception


class SubprocessHandler(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wait_or_raise(self, timeout: Optional[float] = None) -> int:
        """
        Wait for child process to terminate. After wait is finished, the
        error reporter will check whether there were any exceptions.
        """
        exit_code = super().wait(timeout)
        if exit_code != 0:
            try_raise_exception(self.pid, exit_code)
        return exit_code

    def is_alive(self) -> bool:
        """
        Returns True if process is running, otherwise returns False.
        """
        return self.poll() is None

    def _get_signal_name_from_return_code(self, return_code: int) -> Optional[str]:
        if return_code < 0:
            return signal.Signals(-return_code).name
        return None
