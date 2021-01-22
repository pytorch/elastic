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
import pprint
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import IO, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from torchelastic.tsm.driver.api import (
    AppDryRunInfo,
    Resource,
    Application,
    AppState,
    NULL_RESOURCE,
    DescribeAppResponse,
    InvalidRunConfigException,
    SchedulerBackend,
    Role,
    RunConfig,
    Scheduler,
    is_terminal,
    macros,
    runopts,
    NONE,
)
from torchelastic.utils.logging import get_logger


log = get_logger()


def make_unique(app_name: str) -> str:
    return f"{app_name}_{str(uuid4()).split('-')[0]}"


NA: str = "<N/A>"


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
    error_file: str

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

    def __init__(self, id: str, log_dir: str):
        self.id = id
        # cfg.get("log_dir")/<session_name>/<app_id> or /tmp/tsm/<session_name>/<app_id>
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

    def _get_error_file(self) -> Optional[str]:
        error_file = None
        min_timestamp = sys.maxsize
        for replicas in self.role_replicas.values():
            for replica in replicas:
                if not os.path.exists(replica.error_file):
                    continue
                mtime = os.path.getmtime(replica.error_file)
                if mtime < min_timestamp:
                    min_timestamp = mtime
                    error_file = replica.error_file
        return error_file

    def get_structured_error_msg(self) -> str:
        error_file = self._get_error_file()
        if not error_file:
            return NONE

        with open(error_file, "r") as f:
            return json.dumps(json.load(f))

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

        def _fmt_io_filename(std_io: Optional[IO]):
            if std_io:
                return std_io.name
            else:
                return "<CONSOLE>"

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
                    "stdout": _fmt_io_filename(replica.stdout),
                    "stderr": _fmt_io_filename(replica.stderr),
                    "error_file": replica.error_file,
                }
                replicas_info.append(replica_info)
            roles_info[role_name] = replicas_info
        app_info = {
            "app_id": self.id,
            "log_dir": self.log_dir,
            "final_state": self.state.name,
            "last_updated": self.last_updated,
            "roles": roles_info,
        }

        info_str = json.dumps(app_info, indent=2)
        with open(os.path.join(self.log_dir, "SUCCESS"), "w") as fp:
            fp.write(info_str)

        log.info(f"Successfully closed app_id: {self.id}.\n{info_str}")

    def __repr__(self):
        role_to_pid = {}
        for (role_name, replicas) in self.role_replicas.items():
            pids = role_to_pid.setdefault(role_name, [])
            for r in replicas:
                pids.append(r.proc.pid)

        return f"{{app_id:{self.id}, state:{self.state}, pid_map:{role_to_pid}}}"


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


@dataclass
class ReplicaParam:
    """
    Holds ``LocalScheduler._popen()``parameters for each replica of the role.
    """

    args: List[str]
    env: Dict[str, str]
    stdout: Optional[str]
    stderr: Optional[str]


@dataclass
class PopenRequest:
    """
    Holds parameters to create a subprocess for each replica of each role
    of an application.
    """

    app_id: AppId
    log_dir: str
    # maps role_name -> List[ReplicaSpec]
    # role_params["trainer"][0] -> holds trainer's 0^th replica (NOT rank!) parameters
    role_params: Dict[RoleName, List[ReplicaParam]]
    # maps role_name -> List[replica_log_dir]
    # role_log_dirs["trainer"][0] -> holds trainer's 0^th replica's log directory path
    role_log_dirs: Dict[RoleName, List[str]]


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
            help="dir to write stdout/stderr log files of replicas",
        )
        return opts

    def _validate(self, app: Application, scheduler: SchedulerBackend) -> None:
        # Skip validation step for local application
        pass

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

    def _popen(
        self, role_name: RoleName, replica_id: int, replica_params: ReplicaParam
    ) -> _LocalReplica:
        """
        Same as ``subprocess.Popen(**popen_kwargs)`` but is able to take ``stdout`` and ``stderr``
        as file name ``str`` rather than a file-like obj.
        """

        stdout_ = self._get_file_io(replica_params.stdout)
        stderr_ = self._get_file_io(replica_params.stderr)

        # inherit parent's env vars since 99.9% of the time we want this behavior
        # just make sure we override the parent's env vars with the user_defined ones
        env = os.environ.copy()
        env.update(replica_params.env)

        error_file = env["TORCHELASTIC_ERROR_FILE"]

        args_pfmt = pprint.pformat(asdict(replica_params), indent=2, width=80)
        log.info(f"Running {role_name} (replica {replica_id}):\n {args_pfmt}")

        proc = subprocess.Popen(
            args=replica_params.args,
            env=env,
            stdout=stdout_,
            stderr=stderr_,
            preexec_fn=_pr_set_pdeathsig,
        )
        return _LocalReplica(
            role_name,
            replica_id,
            proc,
            stdout=stdout_,
            stderr=stderr_,
            error_file=error_file,
        )

    def _get_app_log_dir(self, app_id: str, cfg: RunConfig) -> Tuple[str, bool]:
        """
        Returns the log dir and a bool (should_redirect_std). We redirect stdout/err
        to a log file ONLY if the log_dir is user-provided in the cfg

        1. if cfg.get("log_dir") -> (user-specified log dir, True)
        2. if not cfg.get("log_dir") -> (autogen tmp log dir, False)
        """

        base_log_dir = cfg.get("log_dir")
        redirect_std = True
        if not base_log_dir:
            base_log_dir = tempfile.mkdtemp(prefix="tsm_")
            redirect_std = False

        return os.path.join(str(base_log_dir), self.session_name, app_id), redirect_std

    def schedule(self, dryrun_info: AppDryRunInfo) -> str:
        if len(self._apps) == self._cache_size:
            if not self._evict_lru():
                raise IndexError(
                    f"App cache size ({self._cache_size}) exceeded. Increase the cache size"
                )

        request: PopenRequest = dryrun_info.request
        app_id = request.app_id
        app_log_dir = request.log_dir
        assert (
            app_id not in self._apps
        ), "no app_id collisons expected since uuid4 suffix is used"

        os.makedirs(app_log_dir)
        local_app = _LocalApplication(app_id, app_log_dir)

        for role_name in request.role_params.keys():
            role_params = request.role_params[role_name]
            role_log_dirs = request.role_log_dirs[role_name]
            for replica_id in range(len(role_params)):
                replica_params = role_params[replica_id]
                replica_log_dir = role_log_dirs[replica_id]

                os.makedirs(replica_log_dir)
                replica = self._popen(role_name, replica_id, replica_params)
                local_app.add_replica(role_name, replica)
        self._apps[app_id] = local_app
        return app_id

    def _submit_dryrun(
        self, app: Application, cfg: RunConfig
    ) -> AppDryRunInfo[PopenRequest]:
        request = self._to_popen_request(app, cfg)
        return AppDryRunInfo(request, lambda p: pprint.pformat(p, indent=2, width=80))

    def _to_popen_request(
        self,
        app: Application,
        cfg: RunConfig,
    ) -> PopenRequest:
        """
        Converts the application and cfg into a ``PopenRequest``.
        """

        app_id = make_unique(app.name)
        image_fetcher = self._get_img_fetcher(cfg)
        app_log_dir, redirect_std = self._get_app_log_dir(app_id, cfg)

        role_params: Dict[str, List[ReplicaParam]] = {}
        role_log_dirs: Dict[str, List[str]] = {}
        for role in app.roles:
            replica_params = role_params.setdefault(role.name, [])
            replica_log_dirs = role_log_dirs.setdefault(role.name, [])

            container = role.container
            img_root = image_fetcher.fetch(container.image)
            cmd = os.path.join(img_root, role.entrypoint)

            for replica_id in range(role.num_replicas):
                args = [cmd] + macros.substitute(
                    role.args, img_root, app_id, str(replica_id)
                )
                replica_log_dir = os.path.join(app_log_dir, role.name, str(replica_id))

                env_vars = {
                    # this is the top level (agent if using elastic role) error file
                    # a.k.a scheduler reply file
                    "TORCHELASTIC_ERROR_FILE": os.path.join(
                        replica_log_dir, "error.json"
                    ),
                    **role.env,
                }
                stdout = None
                stderr = None
                if redirect_std:
                    stdout = os.path.join(replica_log_dir, "stdout.log")
                    stderr = os.path.join(replica_log_dir, "stderr.log")

                replica_params.append(ReplicaParam(args, env_vars, stdout, stderr))
                replica_log_dirs.append(replica_log_dir)

        return PopenRequest(app_id, app_log_dir, role_params, role_log_dirs)

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        if app_id not in self._apps:
            return None

        local_app = self._apps[app_id]
        structured_error_msg = local_app.get_structured_error_msg()

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
        resp.structured_error_msg = structured_error_msg
        resp.state = state
        resp.num_restarts = 0
        resp.ui_url = f"file://{local_app.log_dir}"
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
        log_file = os.path.join(app.log_dir, role_name, str(k), "stderr.log")

        if not os.path.isfile(log_file):
            raise RuntimeError(
                f"app: {app_id} was not configured to log into a file."
                f" Did you run it with log_dir set in RunConfig?"
            )

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
