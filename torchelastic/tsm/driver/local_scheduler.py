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
from uuid import uuid4

from torchelastic.tsm.driver.api import (
    AppDryRunInfo,
    Application,
    AppState,
    DescribeAppResponse,
    InvalidRunConfigException,
    Role,
    RunConfig,
    Scheduler,
    is_terminal,
    macros,
    runopts,
)
from torchelastic.utils.logging import get_logger


log = get_logger()


def make_unique(app_name: str) -> str:
    return f"{app_name}_{str(uuid4()).split('-')[0]}"


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

    def add_process(self, role_name: str, proc: subprocess.Popen) -> None:
        procs = self.role_procs.setdefault(role_name, [])
        procs.append(proc)

    def set_state(self, state: AppState):
        self.last_updated = time.time()
        self.state = state

    def __repr__(self):
        role_to_pid = {}
        for (role_name, procs) in self.role_procs.items():
            pids = role_to_pid.setdefault(role_name, [])
            for p in procs:
                pids.append(p.pid)

        return f"{{name:{self.name}, state:{self.state}, pid_map:{role_to_pid}}}"


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

    def __init__(self, cache_size: int = 100):
        # TODO T72035686 replace dict with a proper LRUCache data structure
        self._apps: Dict[AppId, _LocalApplication] = {}

        if cache_size <= 0:
            raise ValueError("cache size must be greater than zero")
        self._cache_size = cache_size

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add("image_fetcher", type_=str, help="image fetcher type", default="dir")
        return opts

    def _img_fetchers(self) -> Dict[str, ImageFetcher]:
        return {"dir": LocalDirectoryImageFetcher()}

    def _get_img_fetcher(self, cfg: RunConfig) -> ImageFetcher:
        img_fetcher_type = cfg.get("image_fetcher")
        fetchers = self._img_fetchers()
        # pyre-ignore [6]: type check already done by runopt.resolve
        img_fetcher = fetchers.get(img_fetcher_type, None)
        if not img_fetcher:
            raise InvalidRunConfigException(
                f"Unsupported image fetcher type: {img_fetcher_type}. Must be one of: {fetchers.keys()}",
                cfg,
                self.run_opts(),
            )
        return img_fetcher

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
            del self._apps[lru_app_id]

            log.debug(f"evicting app: {lru_app_id}, from local scheduler cache")
            return True
        else:
            log.debug(f"no apps evicted, all {len(self._apps)} apps are running")
            return False

    def _submit(self, app: Application, cfg: RunConfig) -> str:
        if len(self._apps) == self._cache_size:
            if not self._evict_lru():
                raise IndexError(
                    f"App cache size ({self._cache_size}) exceeded. Increase the cache size"
                )

        app_id = make_unique(app.name)
        assert (
            app_id not in self._apps
        ), "no app_id collisons expected since uuid4 suffix is used"

        local_app = _LocalApplication(app.name)

        for role_popen_args in self._to_app_popen_args(app_id, app.roles, cfg):
            for i, (role_name, replica_popen_args) in enumerate(
                role_popen_args.items()
            ):
                for replica_popen_arg in replica_popen_args:
                    log.info(f"Running {role_name} replica {i}): {replica_popen_arg}")
                    proc = subprocess.Popen(**replica_popen_arg)
                    local_app.add_process(role_name, proc)

        self._apps[app_id] = local_app
        return app_id

    def _submit_dryrun(self, app: Application, cfg: RunConfig) -> AppDryRunInfo:
        app_popen_args = self._to_app_popen_args(f"{app.name}_##", app.roles, cfg)
        import pprint

        return AppDryRunInfo(
            app_popen_args, lambda p: pprint.pformat(p, indent=2, width=80)
        )

    def _to_app_popen_args(self, app_id: str, roles: List[Role], cfg: RunConfig):
        """
        returns the popen args for all processes that needs to be created for the app

        ::

         # for each role
         [
           { <role_name_1> : [{args: cmd, env: env, ... other popen args ...}, ...]},
           { <role_name_1> : [{args: cmd, env: env, ... other popen args ...}, ...]},
           ...
         ]

         # example (app has 2 roles: master (1 replica), trainer (2 replicas)
         [
           {
             "master" : [
               {args: "master.par", env: env, ... other popen args ...}
              ]
           },
           {
             "trainer" : [
               {args: "trainer.par", env: env, ... other popen args ...},
               {args: "trainer.par", env: env, ... other popen args ...}
              ]
           },
         ]
        """
        app_popen_params = []
        for role in roles:
            container = role.container
            assert (
                container
            ), "all roles in a submitted app must have container association"

            image_fetcher = self._get_img_fetcher(cfg)
            img_root = image_fetcher.fetch(container.image)
            cmd = os.path.join(img_root, role.entrypoint)

            role_popen_params = {}
            for replica_id in range(role.num_replicas):
                args = [cmd] + macros.substitute(
                    role.args, img_root, app_id, str(replica_id)
                )

                replica_popen_params = role_popen_params.setdefault(role.name, [])
                replica_popen_params.append({"args": args, "env": role.env})
            app_popen_params.append(role_popen_params)
        return app_popen_params

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

    def wait(self, app_id: str) -> Optional[DescribeAppResponse]:
        """
        Waits for the app to finish or raise TimeoutError upon timeout (in seconds).
        If no timeout is specified waits indefinitely.

        Returns:
            The last return value from ``describe()``
        """

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
        # terminate all apps
        for (app_id, app) in self._apps.items():
            log.info(f"Terminating app: {app_id}")
            self._cancel_existing(app_id)


def create_scheduler(**kwargs) -> LocalScheduler:
    return LocalScheduler(cache_size=kwargs.get("cache_size", 100))
