#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import abc
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from torchelastic.tsm.driver.api import (
    Application,
    AppState,
    DescribeAppResponse,
    RunMode,
    Scheduler,
    is_terminal,
    macros,
)
from torchelastic.utils.logging import get_logger


log = get_logger()


class ImageFetcher(abc.ABC):
    """
    Downloads and sets up an image onto the localhost. This is only needed for
    ``LocalhostScheduler`` since typically real schedulers will do this
    on-behalf of the user.
    """

    @abc.abstractmethod
    def fetch(self, image: str) -> str:
        """
        Pulls the given image and returns a path to the pulled image on
        the local host.
        """
        raise NotImplementedError()


class LocalDirectoryImageFetcher(ImageFetcher):
    """
    Interprets the image name as the path to a directory on
    local host. Does not "fetch" (e.g. download) anything. Used in conjunction
    with ``LocalScheduler`` to run local binaries.

    The image name must be an absolute path and must exist.

    Example:

    1. ``fetch(Image(name="/tmp/foobar"))`` returns ``/tmp/foobar``
    2. ``fetch(Image(name="foobar"))`` raises ``ValueError``
    2. ``fetch(Image(name="/tmp/dir/that/does/not_exist"))`` raises ``ValueError``
    """

    def fetch(self, image: str) -> str:
        """
        Raises:
            ValueError - if the image name is not an absolute dir
                         and if it does not exist or is not a directory

        """
        if not os.path.isabs(image):
            raise ValueError(
                f"Invalid image name: {image}, image name must be an absolute path"
            )

        if not os.path.isdir(image):
            raise ValueError(
                f"Invalid image name: {image}, does not exist or is not a directory"
            )

        return image


# aliases to make clear what the mappings are
AppId = str
AppName = str
RoleName = str


class _LocalApplication:
    """
    Container object used by ``LocalhostScheduler`` to group the pids that
    form an application. Each replica of a role in the application is a
    process and has a pid.
    """

    def __init__(self, name: str):
        self.name = name
        # role name -> proc_1, proc_2, ... (proc for each replica)
        self.role_procs: Dict[RoleName, List[subprocess.Popen]] = {}
        self.state: AppState = AppState.PENDING
        # time (in seconds since epoch) when the last set_state method() was called
        self.last_updated: float = -1
        self.run_mode: RunMode = RunMode.MANAGED

    def add_process(self, role_name: str, proc: subprocess.Popen) -> None:
        procs = self.role_procs.setdefault(role_name, [])
        procs.append(proc)

    def set_state(self, state: AppState):
        self.last_updated = time.time()
        self.state = state

    def set_run_mode(self, mode: RunMode):
        self.run_mode = mode

    def __repr__(self):
        role_to_pid = {}
        for (role_name, procs) in self.role_procs.items():
            pids = role_to_pid.setdefault(role_name, [])
            for p in procs:
                pids.append(p.pid)

        return f"{{name:{self.name}, state:{self.state}, mode:{self.run_mode}, pid_map:{role_to_pid}}}"


class LocalScheduler(Scheduler):
    """
    Schedules on localhost. Containers are modeled as processes and
    certain properties of the container that are either not relevant
    or that cannot be enforced for localhost
    runs are ignored. Properties that are ignored:

    1. Resource requirements
    2. Container limit enforcements
    3. Restart policies
    4. Retry counts (no retries supported)

    ..note:: Use this scheduler sparingly since an application
             that runs successfully on a session backed by this
             scheduler may not work on an actual production cluster
             using a different scheduler.
    """

    def __init__(self, image_fetcher: ImageFetcher, cache_size: int = 100):
        # for each app name keeps an incrementing id, to support running
        # multiple instances of the same app
        self._ids: Dict[AppName, int] = {}
        # TODO T72035686 replace dict with a proper LRUCache data structure
        # and don't forget to remove the app name from "_ids" when evicting finished apps=
        self._apps: Dict[AppId, _LocalApplication] = {}
        self._image_fetcher = image_fetcher

        if cache_size <= 0:
            raise ValueError("cache size must be greater than zero")
        self._cache_size = cache_size

    def _evict_lru(self) -> bool:
        """
        Evicts one least recently used element from the apps cache. LRU is defined as
        the oldest app in a terminal state (e.g. oldest finished app).

        Returns:
            ``True`` if an entry was evicted, ``False`` if no entries could be evicted
            (e.g. all apps are running)
        """
        lru_time = sys.maxsize
        lru_app_id = None
        for (app_id, app) in self._apps.items():
            if is_terminal(app.state):
                if app.last_updated <= lru_time:
                    lru_app_id = app_id

        if lru_app_id:
            # evict LRU finished app from the apps cache
            # do not remove the app name from the ids map so that the ids
            # remain unique throughout the lifespan of this scheduler
            # for example if cache size == 1
            #     app_id1 = submit(app)
            #     app_id2 = submit(app) # app_id1 was evicted here
            #     app_id1 == "app.name_0"
            #     app_id2 == "app.name_1"
            del self._apps[lru_app_id]

            log.debug(f"evicting app: {lru_app_id}, from local scheduler cache")
            return True
        else:
            log.debug(f"no apps evicted, all {len(self._apps)} apps are running")
            return False

    def submit(self, app: Application, mode: RunMode) -> str:
        if len(self._apps) == self._cache_size:
            if not self._evict_lru():
                raise IndexError(
                    f"App cache size ({self._cache_size}) exceeded. Increase the cache size"
                )

        id = self._ids.setdefault(app.name, -1) + 1
        self._ids[app.name] = id
        app_id = f"{app.name}_{id}"

        assert (
            app_id not in self._apps
        ), "no app_id collisons expected since incremental integer suffix is used"

        local_app = _LocalApplication(app.name)
        local_app.set_run_mode(mode)

        for role in app.roles:
            container = role.container
            assert (
                container
            ), "all roles in a submitted app must have container association"

            img_root = self._image_fetcher.fetch(container.image)
            cmd = os.path.join(img_root, role.entrypoint)
            for replica_id in range(role.num_replicas):
                args = [cmd] + macros.substitute(
                    role.args, img_root, app_id, str(replica_id)
                )
                log.info(f"Running {args} with env: {role.env}")
                proc = subprocess.Popen(args, env=role.env)
                local_app.add_process(role.name, proc)

        self._apps[app_id] = local_app
        return app_id

    def _failed(self, p: subprocess.Popen):
        if self._is_alive(p):
            return False
        else:
            return p.returncode != 0

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        if app_id not in self._apps:
            return None

        local_app = self._apps[app_id]

        # check if the app has been known to have finished
        if is_terminal(local_app.state):
            state = local_app.state
        else:
            running = False
            failed = False
            for (_, procs) in local_app.role_procs.items():
                for p in procs:
                    running |= self._is_alive(p)
                    failed |= self._failed(p)

            if running:
                state = AppState.RUNNING
            elif failed:
                state = AppState.FAILED
                self._terminate(local_app)  # terminate danglers
            else:
                state = AppState.SUCCEEDED
            local_app.set_state(state)

        resp = DescribeAppResponse()
        resp.app_id = app_id
        resp.state = state
        resp.num_restarts = 0
        return resp

    def _is_alive(self, p: subprocess.Popen):
        return p.poll() is None

    def _terminate(self, app: _LocalApplication):
        for (_, procs) in app.role_procs.items():
            for p in procs:
                # safe to call terminate on a process that already died
                p.terminate()
                p.wait()

    def _cancel_existing(self, app_id: str) -> None:
        # can assume app_id exists
        local_app = self._apps[app_id]
        self._terminate(local_app)
        local_app.state = AppState.CANCELLED

    def wait(self, app_id: str, timeout=-1) -> Optional[DescribeAppResponse]:
        """
        Waits for the app to finish or raise TimeoutError upon timeout (in seconds).
        If no timeout is specified waits indefinitely.

        Returns:
            The last return value from ``describe()``
        """

        if timeout > 0:
            expiry = time.time()
            interval = timeout / 10
        else:
            expiry = sys.maxsize
            interval = 1

        while expiry > time.time():
            desc = self.describe(app_id)

            if desc is None:
                return None
            elif is_terminal(desc.state):
                return desc

            time.sleep(interval)

        raise TimeoutError(f"timed out waiting for app: {app_id} to finish")

    def __del__(self):
        # terminate all MANAGED apps
        for (app_id, app) in self._apps.items():
            if app.run_mode == RunMode.MANAGED:
                log.info(f"Terminating managed app: {app_id}")
                self._cancel_existing(app_id)


def create_scheduler(**kwargs) -> LocalScheduler:
    image_fetcher = LocalDirectoryImageFetcher()
    cache_size = kwargs.get("cache_size", 100)
    return LocalScheduler(image_fetcher, cache_size=cache_size)
