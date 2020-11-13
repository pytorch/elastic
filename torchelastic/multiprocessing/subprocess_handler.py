#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import subprocess
from typing import Optional

from torchelastic.multiprocessing.errors import ProcessFailure, get_failed_result


class SubprocessHandler(subprocess.Popen):
    def __init__(self, rank: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

    def wait_with_return(
        self, timeout: Optional[float] = None, run_id: int = 0
    ) -> Optional[ProcessFailure]:
        """
        Wait for child process to terminate. If the process is finshied
        tries to retrieve the failure.
        """
        exit_code = super().wait(timeout)
        if exit_code != 0:
            return get_failed_result(self.rank, self.pid, exit_code, run_id)
        return None

    def is_alive(self) -> bool:
        """
        Returns True if process is running, otherwise returns False.
        """
        return self.poll() is None
