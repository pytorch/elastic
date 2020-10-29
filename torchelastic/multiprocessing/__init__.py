#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Library that launches multiple processes via a function or a binary entrypoint.
It uses torch.multiprocessing to launch functions and
python subprocess module to launch binaries.

The logical structure of the module and how it relates to torch.mp and subprocess

.. code-block::

    -torchelastic.multiprocessing
     |- api.py # common interfaces that are used by implementations
     |- mp.py # implementations using torch.multiprocessing
     |- sp.py # implementations using python subprocess

Usage 1 (multiprocessing)

::

    # launches 4 processes that execute dummy_fn function and waits for results.

    from torchelastic.multiprocessing import MpParameters

    params = [MpParameters(fn=dummy_fn, args=())]*4
    context = start_processes(start_method = "spawn", *params)
    res_dict = context.wait()


    # launches 4 subprocesses and redirects outputs to pipe

    import subprocess
    from torchelastic.multiprocessing import SubprocessParameters

    params = [SubprocessParameters(args=["ls", "-la", "./"], stdout=subprocess.PIPE)]*4

    context = start_subprocesses(*params)
    context.wait()


"""

from typing import List

import torchelastic.multiprocessing.mp as mp_context
import torchelastic.multiprocessing.sp as sp_context
from torchelastic.multiprocessing.api import BaseProcessContext  # noqa F401
from torchelastic.multiprocessing.mp import MpParameters, MpProcessContext  # noqa F401
from torchelastic.multiprocessing.sp import (  # noqa F401
    SubprocessContext,
    SubprocessParameters,
)


def start_processes(
    params: List[MpParameters],
    start_method: str = "spawn",
):
    """
    Starts processes using torch.multiprocessing.spawn. Each process executes the same
    function. Returns the process context that contains methods over a set of processes.
    Note: All params must have the same values
    """
    proc_params = list(params)
    if len(proc_params) == 0:
        raise ValueError(
            "Params cannot be empty. Provide at least single MpParameters object"
        )
    return mp_context.start_processes(proc_params, start_method)


def start_subprocesses(
    params: List[SubprocessParameters],
):
    """
    Starts processes via subprocess.Popen.
    Returns the process context that contains methods over a set of processes.
    """
    proc_params = list(params)
    if len(proc_params) == 0:
        raise ValueError(
            "Params cannot be empty. Provide at least single SubprocessParameters object"
        )

    return sp_context.start_processes(proc_params)
