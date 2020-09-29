#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import subprocess
from typing import Optional

import torchelastic.multiprocessing.error_reporter as error_reporter


class ResponsivePopen(subprocess.Popen):
    """
    Wait for child process to terminate. After wait is finished, the
    error reporter will check whether there were any exceptions.
    """

    def __init__(self, non_python=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.non_python = non_python

    def wait_or_raise(self, timeout: Optional[float] = None):
        return_code = super().wait(timeout)
        error_message = error_reporter.get_error(self.pid)

        if error_message:
            raise Exception(error_message)
        return return_code
