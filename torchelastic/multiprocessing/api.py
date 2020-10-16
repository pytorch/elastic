#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import signal
from subprocess import CompletedProcess
from typing import List, Optional

from torchelastic.multiprocessing.popen import ResponsivePopen
from torchelastic.multiprocessing.proc_context import (  # noqa F401
    Params,
    ProcContext,
    ProcessGroupException,
    TerminationBehavior,
)


def _pr_set_pdeathsig(sig=signal.SIGTERM):
    """
    Sets PR_SET_PDEATHSIG to ensure a child process is
    terminated appropriately.

    See http://stackoverflow.com/questions/1884941/ for more information.
    For libc.so.6 read http://www.linux-m68k.org/faq/glibcinfo.html
    """
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, sig)


def _process_preexec_fn() -> None:
    _pr_set_pdeathsig()


def run_async(
    params: List[Params],
    termination: TerminationBehavior = TerminationBehavior.GROUP,
    termination_timeout: float = 5,
) -> ProcContext:
    processes = []
    for param in params:
        kwargs = {"args": param.args, "stdout": param.stdout, "stderr": param.stderr}
        if param.env:
            kwargs["env"] = param.env
        kwargs.update(param.kwargs)
        kwargs["preexec_fn"] = _process_preexec_fn
        process = ResponsivePopen(**kwargs)
        processes.append(process)
    return ProcContext(
        processes,
        termination_behavior=termination,
        termination_timeout=termination_timeout,
    )


def run(
    params: List[Params],
    timeout: Optional[float] = None,
    termination_timeout: float = 5,
    termination: TerminationBehavior = TerminationBehavior.GROUP,
) -> Optional[List[CompletedProcess]]:
    proc_context = run_async(
        params=params, termination=termination, termination_timeout=termination_timeout
    )
    return proc_context.wait(timeout)
