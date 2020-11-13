#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import signal
import subprocess
from typing import Optional, Tuple

from torchelastic.multiprocessing.errors import ProcessFailure, get_failed_result


class SubprocessHandler:
    def __init__(self, rank: int = 0, *args, **kwargs):
        self._popen = subprocess.Popen(*args, **kwargs)
        self.rank = rank

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def args(self) -> Tuple:
        return self._popen.args

    @property
    def stdout(self):
        return self._popen.stdout

    @property
    def returncode(self):
        return self._popen.returncode

    @property
    def stderr(self):
        return self._popen.stderr

    def wait(self, timeout: Optional[float] = None) -> int:
        """
        Waits for process to terminate. The method is using  ``subprocess.Popen.wait``.
        """
        return self._popen.wait(timeout)

    def terminate(self) -> None:
        """
        Terminates the process. The method is using  ``subprocess.Popen.terminate``.
        """
        self._popen.terminate()

    def wait_with_return(
        self, timeout: Optional[float] = None, run_id: int = 0
    ) -> Optional[ProcessFailure]:
        """
        Wait for child process to terminate. If the process is finshied
        tries to retrieve the failure.
        """
        exit_code = self._popen.wait(timeout)
        if exit_code != 0:
            return get_failed_result(self.rank, self.pid, exit_code, run_id)
        return None

    def is_alive(self) -> bool:
        """
        Returns True if process is running, otherwise returns False.
        """
        return self._popen.poll() is None

    def _get_signal_name_from_return_code(self, return_code: int) -> Optional[str]:
        if return_code < 0:
            return signal.Signals(-return_code).name
        return None
