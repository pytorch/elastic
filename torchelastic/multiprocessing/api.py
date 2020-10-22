#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Dict, List, Optional


class BaseProcessContext(abc.ABC):
    """
    The base class that standardizes operations over a set of processes
    that are launched via different mechanisms.
    """

    @abc.abstractmethod
    def wait(self, timeout: Optional[float] = None) -> Optional[Dict[int, Any]]:
        r"""
        Waits for processes to finish. If timeout is not specified, the method will block
        until all processes are finished.
        Returns None if any processes are running. Otherwise returns a dict of local_rank and output,
        where local_rank is in range [0,nprocs].
        The method should support all-or-nothing policy, meaning that it will retun ``Dict[int, Any]``
        only if all processes are finished successfully. If any processes are running, the method will
        return None. If any process fails, the method will throw Exception and terminate the other processes
        via SIGTERM.

        Arguments:
            timeout (Optional[float]): the time to wait for processes to finish. The value None means
                that the method will block until all procsses finish or any of them fail. The timeout=-1
                means that the method will return immediatelly without waiting.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def pids(self) -> List[int]:
        """
        Returns pids of processes.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def terminate(self) -> None:
        r"""
        Terminates all processes.
        """
        raise NotImplementedError()
