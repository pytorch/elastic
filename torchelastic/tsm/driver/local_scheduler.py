#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import abc
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any, Dict, Iterable, List, Optional
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


@dataclass
class _LocalReplica:
    """
    Contains information about a locally running role replica.
    """

    role_name: RoleName
    replica_id: int
    proc: subprocess.Popen
    stdout: Optional[IO]  # None means no log_dir (out to console)
    stderr: Optional[IO]  # None means no log_dir (out to console)

    def terminate(self) -> None:
        """
        terminates the underlying process for this replica
        closes stdout and stderr file handles
        safe to call multiple times
        """
        # safe to call terminate on a process that already died
        self.proc.terminate()
        self.proc.wait()

        # close stdout and stderr log file handles
        if self.stdout:
            # pyre-ignore [16] already null checked
            self.stdout.close()
        if self.stderr:
            self.stderr.close()

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def failed(self) -> bool:
        if self.is_alive():  # if still running, then has not failed
            return False
        else:
            return self.proc.returncode != 0


class _LocalApplication:
    """
    Container object used by ``LocalhostScheduler`` to group the pids that
    form an application. Each replica of a role in the application is a
    process and has a pid.
    """

    def __init__(self, name: str, id: str, log_dir: Optional[str]):
        self.name = name
        self.id = id
        # None or cfg.get("log_dir")/<session_name>/<app_id>
        self.log_dir = log_dir
        # role name -> [replicas, ...]
        self.role_replicas: Dict[RoleName, List[_LocalReplica]] = {}
        self.state: AppState = AppState.PENDING
        # time (in seconds since epoch) when the last set_state method() was called
        self.last_updated: float = -1

    def add_replica(self, role_name: str, replica: _LocalReplica) -> None:
        procs = self.role_replicas.setdefault(role_name, [])
        procs.append(replica)

    def set_state(self, state: AppState) -> None:
        self.last_updated = time.time()
        self.state = state

    def terminate(self) -> None:
        """
        terminates all procs associated with this app,
        and closes any resources (e.g. log file handles)
        safe to call multiple times
        """
        # terminate all replica processes
        for replicas in self.role_replicas.values():
            for r in replicas:
                r.terminate()

    def close(self) -> None:
        """
        terminates all procs associated with this app,
        and closes any resources (e.g. log file handles)
        and if log_dir has been specified,
        writes a SUCCESS file indicating that the log files
        have been flushed and closed and ready to read.
        NOT safe to call multiple times!
        """
        self.terminate()

        # drop a SUCCESS file in the log dir to signal that
        # the log file handles have all been closed properly
        # and that they can reliably be read
        roles_info = {}
        for role_name, replicas in self.role_replicas.items():
            replicas_info = []
            for replica in replicas:
                replica_info = {
                    "replica_id": replica.replica_id,
                    "pid": replica.proc.pid,
                    "exitcode": replica.proc.returncode,
                }
                if self.log_dir:
                    # pyre-ignore [16] replica.stdout|stderr is a file handle if log_dir
                    replica_info["stdout"] = replica.stdout.name
                    replica_info["stderr"] = replica.stderr.name
                replicas_info.append(replica_info)
            roles_info[role_name] = replicas_info
        app_info = {
            "app_name": self.name,
            "app_id": self.id,
            "log_dir": self.log_dir or "<None>",
            "final_state": self.state.name,
            "last_updated": self.last_updated,
            "roles": roles_info,
        }
        info_str = json.dumps(app_info, indent=2)

        if self.log_dir:
            # pyre-ignore [6]: app.log_dir nullchecked above
            with open(os.path.join(self.log_dir, "SUCCESS"), "w") as fp:
                fp.write(info_str)

        log.info(f"Successfully closed app_id: {self.id}.\n{info_str}")

    def __repr__(self):
        role_to_pid = {}
        for (role_name, replicas) in self.role_replicas.items():
            pids = role_to_pid.setdefault(role_name, [])
            for r in replicas:
                pids.append(r.proc.pid)

        return f"{{name:{self.name}, state:{self.state}, pid_map:{role_to_pid}}}"


def _pr_set_pdeathsig() -> None:
    """
    Sets PR_SET_PDEATHSIG to ensure a child process is
    terminated appropriately.

    See http://stackoverflow.com/questions/1884941/ for more information.
    For libc.so.6 read http://www.linux-m68k.org/faq/glibcinfo.html
    """
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


class LocalScheduler(Scheduler):
    """
    Schedules on localhost. Containers are modeled as processes and
    certain properties of the container that are either not relevant
    or that cannot be enforced for localhost
    runs are ignored. Properties that are ignored:

    1. Resource requirements
    2. Container limit enforcements
    3. Retry policies
    4. Retry counts (no retries supported)
    5. Deployment preferences

    ..note:: Use this scheduler sparingly since an application
             that runs successfully on a session backed by this
             scheduler may not work on an actual production cluster
             using a different scheduler.
    """

    def __init__(self, session_name: str, cache_size: int = 100):
        super().__init__(session_name)

        # TODO T72035686 replace dict with a proper LRUCache data structure
        self._apps: Dict[AppId, _LocalApplication] = {}

        if cache_size <= 0:
            raise ValueError("cache size must be greater than zero")
        self._cache_size = cache_size

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add("image_fetcher", type_=str, help="image fetcher type", default="dir")
        opts.add(
            "log_dir",
            type_=str,
            default=None,
            help="dir to write stdout/stderr log files of replicas (defaults to stdout|err PIPE)",
        )
        return opts

    def _img_fetchers(self) -> Dict[str, ImageFetcher]:
        return {"dir": LocalDirectoryImageFetcher()}

    def _get_app_log_dir(self, app_id: str, cfg: RunConfig) -> Optional[str]:
        # pyre-ignore [6]: type check already done by runopt.resolve
        log_dir: str = cfg.get("log_dir")
        if log_dir:
            return os.path.join(log_dir, self.session_name, app_id)
        else:
            return None

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

    def _get_file_io(self, file: Optional[str]):
        """
        Given a file name, opens the file for write and returns the IO.
        If no file name is given, then returns ``None``
        Raises a ``FileExistsError`` if the file is already present.
        """

        if not file:
            return None

        if os.path.isfile(file):
            raise FileExistsError(
                f"log file: {file} already exists,"
                f" specify a different log_dir, app_name, or remove the file and retry"
            )

        os.makedirs(os.path.dirname(file), exist_ok=True)
        return open(file, mode="w")

    def _popen(self, role_name: str, replica_id: int, **popen_kwargs) -> _LocalReplica:
        """
        Same as ``subprocess.Popen(**popen_kwargs)`` but is able to take ``stdout`` and ``stderr``
        as file name ``str`` rather than a file-like obj.
        """
        stdout_ = self._get_file_io(popen_kwargs.pop("stdout", None))
        stderr_ = self._get_file_io(popen_kwargs.pop("stderr", None))

        proc = subprocess.Popen(
            **popen_kwargs, stdout=stdout_, stderr=stderr_, preexec_fn=_pr_set_pdeathsig
        )
        return _LocalReplica(
            role_name, replica_id, proc, stdout=stdout_, stderr=stderr_
        )

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

        local_app = _LocalApplication(
            app.name, app_id, self._get_app_log_dir(app_id, cfg)
        )

        for role_popen_args in self._to_app_popen_args(app_id, app.roles, cfg):
            for role_name, replica_popen_args in role_popen_args.items():
                for replica_id, replica_popen_arg in enumerate(replica_popen_args):
                    log.info(f"Running {role_name} ({replica_id}): {replica_popen_arg}")
                    replica = self._popen(role_name, replica_id, **replica_popen_arg)
                    local_app.add_replica(role_name, replica)

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
                params: Dict[str, Any] = {"args": args, "env": role.env}
                app_log_dir = self._get_app_log_dir(app_id, cfg)
                if app_log_dir:
                    base_log_dir = os.path.join(app_log_dir, role.name, str(replica_id))
                    params["stdout"] = os.path.join(base_log_dir, "stdout.log")
                    params["stderr"] = os.path.join(base_log_dir, "stderr.log")

                replica_popen_params.append(params)

            app_popen_params.append(role_popen_params)
        return app_popen_params

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        if app_id not in self._apps:
            return None

        local_app = self._apps[app_id]

        # check if the app is known to have finished
        if is_terminal(local_app.state):
            state = local_app.state
        else:
            running = False
            failed = False
            for replicas in local_app.role_replicas.values():
                for r in replicas:
                    running |= r.is_alive()
                    failed |= r.failed()

            if running:
                state = AppState.RUNNING
            elif failed:
                state = AppState.FAILED
            else:
                state = AppState.SUCCEEDED
            local_app.set_state(state)

        if is_terminal(local_app.state):
            local_app.close()

        resp = DescribeAppResponse()
        resp.app_id = app_id
        resp.state = state
        resp.num_restarts = 0
        return resp

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterable:
        if since or until:
            warnings.warn(
                "Since and/or until times specified for LocalScheduler.log_iter."
                " These will be ignored and all log lines will be returned"
            )

        app = self._apps[app_id]
        if not app.log_dir:
            raise RuntimeError(
                f"app: {app_id} was not configured to log into a file."
                f" Did you run it with log_dir set in RunConfig?"
            )

        # pyre-ignore [6]: app.log_dir nullchecked above
        log_file = os.path.join(app.log_dir, role_name, str(k), "stderr.log")
        return LogIterator(app_id, regex or ".*", log_file, self)

    def _cancel_existing(self, app_id: str) -> None:
        # can assume app_id exists
        local_app = self._apps[app_id]
        local_app.close()
        local_app.state = AppState.CANCELLED

    def __del__(self):
        # terminate all apps
        for (app_id, app) in self._apps.items():
            log.info(f"Terminating app: {app_id}")
            app.terminate()


class LogIterator:
    def __init__(
        self, app_id: str, regex: str, log_file: str, scheduler: LocalScheduler
    ):
        self._app_id: str = app_id
        self._regex = re.compile(regex)
        self._log_file: str = log_file
        self._log_fp: Optional[IO] = None
        self._scheduler: LocalScheduler = scheduler
        self._app_finished: bool = False

    def _check_finished(self):
        # either the app (already finished) was evicted from the LRU cache
        # -- or -- the app reached a terminal state (and still in the cache)
        desc = self._scheduler.describe(self._app_id)
        if not desc or is_terminal(desc.state):
            self._app_finished = True
        else:
            self._app_finished = False

    def __iter__(self):
        # wait for the log file to appear or app to finish (whichever happens first)
        while True:
            self._check_finished()  # check to see if app has finished running

            if os.path.isfile(self._log_file):
                self._log_fp = open(self._log_file, "r")  # noqa: P201
                break

            if self._app_finished:
                # app finished without ever writing a log file
                raise RuntimeError(
                    f"app: {self._app_id} finished without writing: {self._log_file}"
                )

            time.sleep(1)
        return self

    def __next__(self):
        while True:
            line = self._log_fp.readline()
            if not line:
                # we have reached EOF and app finished
                if self._app_finished:
                    self._log_fp.close()
                    raise StopIteration()

                # if app is still running we need to wait for more possible log lines
                # sleep for 1 sec to avoid thrashing the follow
                time.sleep(1)
                self._check_finished()
            else:
                line = line.rstrip("\n")  # strip the trailing newline
                if re.match(self._regex, line):
                    return line


def create_scheduler(session_name: str, **kwargs) -> LocalScheduler:
    return LocalScheduler(
        session_name=session_name,
        cache_size=kwargs.get("cache_size", 100),
    )
