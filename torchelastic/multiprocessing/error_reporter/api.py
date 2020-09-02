#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

import logging
import os
from typing import Any, Callable, Optional, Tuple


log: logging.Logger = logging.getLogger(__name__)


_signal_handler_dict = {}
_ERROR_FILE_ENV: str = "ERROR_FILE_ENV_VAR"
_session_identifier_env_var: str = "SESSION_IDENTIFIER_ENV_VAR"


def configure(session_identifier: str) -> None:
    # TODO: lazy init this in the __init__.py file, in case user does not call configure.
    """
    Configure error reporter
    session_identifier: a unique identifier for each error reporting session,
        to guarantee unique error file paths
        exp: timestamp for each multiprocessing.spawn call
    """
    # TODO(T74327900) : make cleanup procedure configurable
    os.environ[_session_identifier_env_var] = session_identifier
    log.info(f"session_id set to {_get_session_identifier()}")


def _get_session_identifier() -> str:
    return os.getenv(_session_identifier_env_var, "")


def _register_signal_handler(signal_handler, platform: Optional[str] = None) -> None:
    log.info(
        f"{os.getpid()} _register_signal_handler {signal_handler} with plaform={platform}"
        f", current platform={get_platform()}"
    )
    if platform is None:
        _signal_handler_dict[get_platform()] = signal_handler
    else:
        _signal_handler_dict[platform] = signal_handler


def exec_fn(user_funct: Callable, args: Tuple = ()) -> Any:
    log.info(f"exec_fn process {os.getpid()}, current platform={get_platform()}")
    signal_intercepter = _signal_handler_dict[get_platform()]
    signal_intercepter()
    return user_funct(*args)


def get_error(error_process_pid: int) -> str:
    pass


def get_platform() -> str:
    """
    Finds what platform this module is currently running on, and returns
    str identifier. Eg. local
    """
    return "LOCAL"


def _local_handler() -> None:
    # TODO (T73940701)
    pass


_register_signal_handler(_local_handler)
