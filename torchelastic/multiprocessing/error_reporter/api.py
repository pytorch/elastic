#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

import logging
import os
import time
from typing import Any, Callable, Optional, Tuple

from torchelastic.multiprocessing.error_reporter.handlers import get_signal_handler


log: logging.Logger = logging.getLogger(__name__)


def configure(sess_identifier: Optional[str] = None) -> None:
    # TODO: lazy init this in the __init__.py file, in case user does not call configure.
    """
    Configure error reporter
    session_identifier: a unique identifier for each error reporting session,
        to guarantee unique error file paths
        exp: timestamp for each multiprocessing.spawn call
    """
    # TODO(T74327900) : make cleanup procedure configurable
    # TODO(avianou) : remove session identifier, since we can use differen mechanism
    # to guarantee path uniquness. Also session identifier lacks debuggability
    if not sess_identifier:
        sess_identifier = str(int(time.monotonic() * 1000))
    os.environ["SESSION_IDENTIFIER_ENV_VAR"] = sess_identifier
    log.info(f"session_id set to {sess_identifier}")


def exec_fn(user_funct: Callable, args: Tuple = ()) -> Any:
    signal_handler = get_signal_handler()
    log.info(
        f"exec_fn process {os.getpid()}, using signal hanler: {signal_handler.__class__}"
    )
    signal_handler.configure()
    return user_funct(*args)


def get_error(error_process_pid: int) -> str:
    signal_handler = get_signal_handler()
    return signal_handler.construct_error_message(error_process_pid)
