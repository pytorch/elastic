#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Dict, List, Set

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer


class LocalTimerClient(TimerClient):
    def __init__(self, mp_queue):
        super().__init__()
        self._mp_queue = mp_queue

    def acquire(self, scope_id, expiration_time):
        pid = os.getpid()
        acquire_request = TimerRequest(pid, scope_id, expiration_time)
        self._mp_queue.put(acquire_request)

    def release(self, scope_id):
        pid = os.getpid()
        release_request = TimerRequest(pid, scope_id, -1)
        self._mp_queue.put(release_request)


class MultiprocessingRequestQueue(RequestQueue):
    """
    A ``RequestQueue`` backed by python ``multiprocessing.Queue``
    """

    def __init__(self, mp_queue: mp.Queue):
        super().__init__()
        self._mp_queue = mp_queue

    def size(self) -> int:
        return self._mp_queue.qsize()

    def get(self, size, timeout: float) -> List[TimerRequest]:
        requests = []
        wait = timeout
        for _ in range(0, size):
            start = time.time()

            try:
                r = self._mp_queue.get(block=True, timeout=wait)
            except Empty:
                break

            requests.append(r)
            wait = wait - (time.time() - start)
            if wait <= 0:
                break

        return requests


class LocalTimerServer(TimerServer):
    def __init__(
        self, mp_queue: mp.Queue, max_interval: float = 60, daemon: bool = True
    ):
        super().__init__(MultiprocessingRequestQueue(mp_queue), max_interval, daemon)
        self._timers = {}

    def register_timers(self, timer_requests: List[TimerRequest]) -> None:
        for request in timer_requests:
            pid = request.worker_id
            scope_id = request.scope_id
            expiration_time = request.expiration_time

            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                self._timers.pop((pid, scope_id), None)
            else:
                self._timers[(pid, scope_id)] = request

    def clear_timers(self, worker_ids: Set[int]) -> None:
        for (pid, scope_id) in list(self._timers.keys()):
            if pid in worker_ids:
                self._timers.pop((pid, scope_id))

    def get_expired_timers(self, deadline: float) -> Dict[str, List[TimerRequest]]:
        # pid -> [timer_requests...]
        expired_timers = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_id, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_id: int) -> bool:
        try:
            os.kill(worker_id, signal.SIGKILL)
            return True
        except ProcessLookupError:
            logging.info(f"Process with pid={worker_id} does not exist. Skipping")
            return True
        except Exception as e:
            logging.error(f"Error terminating pid={worker_id}", exc_info=e)
        return False
