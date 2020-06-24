#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
from typing import Optional


def get_logger(name: Optional[str] = None):
    r"""
    Util function to set up a simple logger that writes
    into stderr. The loglevel is fetched from the LOGLEVEL
    env. variable or INFO as default. The function will use the
    module name of the caller if no name is provided.

    Arguments:
        name (str): Name of the logger. If no name provided, the name will
        be derived from the call stack.
    """

    if name is None:
        try:
            # Derive the name of the caller. Since the module name derivation
            # is a function we use the depth=2 to find the caller.
            name = _derive_module_name(depth=2)
        except Exception as e:
            default_log = _setup_logger()
            default_log.warn(
                f"Error while setting up logger. Will be using default logger. Got exception: {e}"
            )

    log = _setup_logger(name)
    return log


def _setup_logger(name: Optional[str] = None):
    log = logging.getLogger(name)
    if len(log.handlers) == 0:
        # Add default handler that writes messages to stderr
        log.addHandler(logging.StreamHandler())
    log.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    return log


def _derive_module_name(depth: int = 1) -> str:
    r"""
    Derives the name of the caller module from the stack frames.

    Arguments:
        depth (int): The position of the frame in the stack.
    """
    stack = inspect.stack()
    assert depth < len(stack)
    node = stack[depth]
    # Each element of the stack is an array of elements, where the
    # first element should alway be a fame object.
    frame = node[0]
    module = inspect.getmodule(frame)
    if module is None:
        raise ValueError(f"Frame {frame} at depth {depth} does not have module.")
    return module.__name__
