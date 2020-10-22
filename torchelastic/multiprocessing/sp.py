#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import os
import signal
import sys
import time
from subprocess import CompletedProcess, TimeoutExpired
from typing import Dict, List, Optional, Tuple, Union

from torchelastic.multiprocessing.api import BaseProcessContext
from torchelastic.multiprocessing.base_process_handler import ProcessException
from torchelastic.multiprocessing.process_handler import ProcessHandler


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


def _preexec_fn() -> None:
    _pr_set_pdeathsig()


class SubprocessParameters:
    """
    Specifies parameters that are used for launching subprocesses.
    The class accepts arbitrary kwargs that will be directly passed to
    the ``subprocess.Popen`` constructor
    """

    __slots__ = ("args", "stdout", "stderr", "start_args")

    def __init__(
        self,
        args: Tuple,
        stdout: Union[int, str, None] = None,
        stderr: Union[int, str, None] = None,
        **start_kwargs,
    ):
        r"""

        Arguments:
            args (Tuple[str]): Arguments that are used to launch the process
            stdout (Union[int, str, None]) The destination of the stdout stream. It can have one
                of three types: None - the default output, int - the fd destination, which also
                can be a special use case, e.g. subprocess.PIPE, str - the directory destination.
                If the value is str, the process will write the output into: {stdout}/{rank}/stdout.log
            stderr (Union[int, str, None]) The destination of the stderr stream. It can have one
                of three types: None - the default output, int - the fd destination, which also
                can be a special use case, e.g. subprocess.PIPE, str - the directory destination.
                If the value is str, the process will write the output into: {stdout}/{rank}/stderr.log
        """
        self.args = args
        self.stdout = stdout
        self.stderr = stderr
        self.start_args = {}
        for (key, val) in start_kwargs.items():
            self.start_args[key] = val


class SubprocessContext(BaseProcessContext):
    """
    Combines common operations that are executed on the same process group.
    """

    def __init__(
        self,
        proc_list: List[ProcessHandler],
    ):
        self.processes = proc_list

    def wait(
        self, timeout: Optional[float] = None
    ) -> Optional[Dict[int, CompletedProcess]]:
        r"""
        Method waits for process group completion in a loop. If all processes succeeded the list of
        ``subprocess.CompletedProcess`` will be returned. If processes are still running the method
        will return None.
        The method will throw the exception that is first got recorded.
        """
        if timeout and timeout > 0:
            deadline = time.time() + timeout
            period = min(1, int(timeout / 10))
        elif timeout and timeout <= 0:
            deadline = time.time() + 1  # wait for one second
            period = 1  # one second
        else:
            deadline = sys.maxsize
            period = 1
        while deadline > time.time():
            if not self._any_alive():
                break
            root_exception = self._try_wait_and_raise()
            if root_exception:
                self.terminate()
                raise root_exception
            time.sleep(period)
        if self._any_alive():
            return None
        root_exception = self._try_wait_and_raise()
        if root_exception:
            raise root_exception
        results = {}
        for idx, proc in enumerate(self.processes):
            results[idx] = CompletedProcess(
                args=proc.args,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        return results

    def pids(self) -> List[int]:
        return [int(proc.pid) for proc in self.processes]

    def _any_alive(self) -> bool:
        for proc in self.processes:
            if proc.is_alive():
                return True
        return False

    def _try_wait_and_raise(self, timeout: float = 1.0) -> Optional[ProcessException]:
        root_exception = None
        for proc in self.processes:
            try:
                proc.wait_or_raise(timeout)
            except TimeoutExpired:
                pass
            except ProcessException as e:
                if not root_exception or root_exception.timestamp > e.timestamp:
                    root_exception = e
        return root_exception

    def terminate(self) -> None:
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
                proc.wait()


def _resolve_std_stream(stream: Union[str, int, None], type: str, rank: int):
    if stream is None:
        return None
    elif isinstance(stream, int):
        return stream
    else:
        path = f"{rank}/{type}.log"
        stream_file = os.path.join(stream, path)
        stream_dir = stream_file[0 : stream_file.rfind(os.path.sep)]
        os.makedirs(stream_dir, exist_ok=True)
        return open(stream_file, "w")


def start_processes(
    params: List[SubprocessParameters],
) -> SubprocessContext:

    processes = []
    for local_rank, proc_params in enumerate(params):
        stdout_stream = _resolve_std_stream(proc_params.stdout, "stdout", local_rank)
        stderr_stream = _resolve_std_stream(proc_params.stderr, "stderr", local_rank)
        popen_args = proc_params.start_args
        proc_args = {
            "args": proc_params.args,
            "preexec_fn": _preexec_fn,
            "stdout": stdout_stream,
            "stderr": stderr_stream,
        }
        popen_args.update(proc_args)
        process = ProcessHandler(**popen_args)
        processes.append(process)
    return SubprocessContext(processes)
