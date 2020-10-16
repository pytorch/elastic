#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import sys
import time
from enum import Enum
from subprocess import CompletedProcess, TimeoutExpired
from typing import Dict, List, Optional

from torchelastic.multiprocessing.popen import ProcessException, ResponsivePopen


class TerminationBehavior(Enum):
    """
    Process group termination behavior:

      GROUP - when the error occurs on a single process, wait for some time for other processes
        to complete, and teminate the rest via SIGTERM.
      SINGLE - raise error immediately without terminating the rest of the group.

    """

    GROUP = 0
    SINGLE = 1


class ProcessGroupException(Exception):
    def __init__(self, msg: str, errors: Dict[int, ProcessException]):
        super().__init__(msg)
        self._errors = errors

    def get_errors(self) -> Dict[int, ProcessException]:
        return self._errors


class Params:
    """
    Represents parameters to ``ResponsivePopen``. Class accepts any parameters that the standard
    ``subprocess.Popen`` accepts, except preexec_fn.
    """

    def __init__(
        self,
        args: List[str],
        stdout=None,
        stderr=None,
        env: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.args = args
        self.stdout = stdout
        self.stderr = stderr
        self.env = env
        self.kwargs = kwargs

    def copy_obj(self) -> "Params":
        return copy.deepcopy(self)

    def replicate(self, times: int = 1) -> List["Params"]:
        return [self.copy_obj() for _ in range(times)]


class ProcContext:
    """
    Combines common operations that are executed on the same process group.
    """

    def __init__(
        self,
        proc_list: List[ResponsivePopen],
        termination_behavior: TerminationBehavior,
        termination_timeout: float = 5,  # seconds
    ):
        self.processes = proc_list
        self.termination_behavior = termination_behavior
        self.termination_timeout = termination_timeout

    def check(self) -> Optional[List[CompletedProcess]]:
        r"""
        Waits for one second and returns the result.
        """
        return self.wait(1)

    def wait(self, timeout: Optional[float] = None) -> Optional[List[CompletedProcess]]:
        r"""
        Method waits for process group completion in a loop. If all processes succeeded the list of
        ``subprocess.CompletedProcess`` will be returned. If processes are still running the method
        will return None. The failure behavior depends on the ``termination_behavior`` parameter.
        If ``termination_behavior`` is ``TerminationBehavior.SINGLE`` method will raise
        ``ProcessGroupException`` that has the first exception that occurred. The method will not
        terminate other running processes.
        If ``termination_behavior`` is ``TerminationBehavior.GROUP`` method will try to wait
        ``termination_timeout`` time and terminate the rest of the processes. The ``ProcessGroupException``
        will be raised.
        """
        if timeout and timeout > 0:
            deadline = time.time() + timeout
            period = min(1, int(timeout / 10))
        else:
            deadline = sys.maxsize
            period = 1
        while deadline > time.time():
            if not self.any_alive():
                break
            for idx, proc in enumerate(self.processes):
                try:
                    proc.wait_or_raise(1)
                except TimeoutExpired:
                    pass
                except ProcessException as e:
                    raised_errors = {idx: e}
                    if self.termination_behavior == TerminationBehavior.GROUP:
                        self._wait_and_terminate(
                            raised_errors, self.termination_timeout
                        )
                    raise ProcessGroupException("Process group failed.", raised_errors)
            time.sleep(period)
        if self.any_alive():
            return None
        raised_errors = {}
        self._wait_and_terminate(raised_errors, 0)
        if len(raised_errors) != 0:
            raise ProcessGroupException("Process group failed.", raised_errors)
        results = []
        for proc in self.processes:
            results.append(
                CompletedProcess(
                    args=proc.args,
                    returncode=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )
            )
        return results

    def any_alive(self) -> bool:
        for proc in self.processes:
            if proc.is_alive():
                return True
        return False

    def _wait_and_terminate(
        self, raised_errors: Dict[int, ProcessException], termination_timeout
    ) -> None:
        time.sleep(termination_timeout)
        self.terminate_all()
        for idx, proc in enumerate(self.processes):
            if idx in raised_errors:
                continue
            try:
                proc.wait_or_raise(1)
            except TimeoutExpired:
                pass
            except ProcessException as e:
                raised_errors[idx] = e

    def terminate_all(self) -> None:
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
                proc.wait()
