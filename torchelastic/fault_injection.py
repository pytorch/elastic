#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import signal
import threading
import time


def start_fault_injection_thread(fault_injection_rate, interval_in_sec):
    """
    Useful tool for validating Elastic Trainer behavior under stress.
    Can be used in production workflows to ensure correct behavior of
    various components, stateful side-effects, etc. (not just for unit tests).
    """

    def _fault_injection_loop():
        while True:
            x = random.random()
            if x < fault_injection_rate:
                logging.error(
                    "Fault injection triggered! roll: {}, probability: {}".format(
                        x, fault_injection_rate
                    )
                )
                os.kill(os.getpid(), signal.SIGKILL)

            time.sleep(interval_in_sec)

    fault_injection_thread = threading.Thread(target=_fault_injection_loop)
    fault_injection_thread.daemon = True
    fault_injection_thread.start()
