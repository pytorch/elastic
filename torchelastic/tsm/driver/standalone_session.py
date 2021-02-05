#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import getpass
import json
import socket
import time
import traceback
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from torchelastic.events import record_tsm, TsmEvent
from torchelastic.tsm.driver.api import (
    AppDryRunInfo,
    AppHandle,
    Application,
    AppStatus,
    RunConfig,
    Scheduler,
    SchedulerBackend,
    Session,
    SessionMismatchException,
    UnknownAppException,
    make_app_handle,
    parse_app_handle,
    runopts,
)


class LoggingSession(Session):
    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)

    def schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        scheduler_backend = dryrun_info._scheduler
        runcfg = json.dumps(dryrun_info._cfg.cfgs) if dryrun_info._cfg else None
        tsm_event = self._generate_tsm_event(
            "schedule",
            scheduler_backend,
            runcfg=runcfg,
        )
        try:
            app_handle = self._schedule(dryrun_info)
            _, _, app_id = parse_app_handle(app_handle)
            tsm_event.app_id = app_id
            # TODO(avivanou): t81936552 each action corresponds to a method call
            # as a result instead of repeatedly log events in each method, we
            # can log them implicitly via footsteps lib
            record_tsm(tsm_event)
            return app_handle
        except Exception:
            tsm_event.raw_exception = traceback.format_exc()
            record_tsm(tsm_event)
            raise

    def status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        # allow status checks of apps from other sessions
        scheduler_backend, _, app_id = parse_app_handle(app_handle)
        tsm_event = self._generate_tsm_event("status", scheduler_backend, app_id)
        try:
            app_status = self._status(app_handle)
            record_tsm(tsm_event)
            return app_status
        except Exception:
            tsm_event.raw_exception = traceback.format_exc()
            record_tsm(tsm_event)
            raise

    def wait(self, app_handle: AppHandle) -> Optional[AppStatus]:
        scheduler_backend, _, app_id = parse_app_handle(app_handle)
        tsm_event = self._generate_tsm_event("wait", scheduler_backend, app_id)
        try:
            record_tsm(tsm_event)
            return self._wait(app_handle)
        except Exception:
            tsm_event.raw_exception = traceback.format_exc()
            record_tsm(tsm_event)
            raise

    def list(self) -> Dict[AppHandle, Application]:
        tsm_event = self._generate_tsm_event("list", "")
        try:
            res = self._list()
            record_tsm(tsm_event)
            return res
        except Exception:
            tsm_event.raw_exception = traceback.format_exc()
            record_tsm(tsm_event)
            raise

    def stop(self, app_handle: AppHandle) -> None:
        scheduler_backend, _, app_id = parse_app_handle(app_handle)
        tsm_event = self._generate_tsm_event(
            "stop",
            scheduler_backend,
            app_id,
        )
        try:
            self._stop(app_handle)
            record_tsm(tsm_event)
        except Exception:
            tsm_event.raw_exception = traceback.format_exc()
            record_tsm(tsm_event)
            raise

    def describe(self, app_handle: AppHandle) -> Optional[Application]:
        scheduler_backend, _, app_id = parse_app_handle(app_handle)

        tsm_event = self._generate_tsm_event(
            "describe",
            scheduler_backend,
            app_id,
        )
        try:
            res = self._describe(app_handle)
            record_tsm(tsm_event)
            return res
        except Exception:
            tsm_event.raw_exception = traceback.format_exc()
            record_tsm(tsm_event)
            raise

    def log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterable:
        scheduler_backend, _, app_id = parse_app_handle(app_handle)
        tsm_event = self._generate_tsm_event(
            "log_lines",
            scheduler_backend,
            app_id,
        )
        try:
            log_iter = self._log_lines(app_handle, role_name, k, regex, since, until)
            record_tsm(tsm_event)
            return log_iter
        except Exception:
            tsm_event.raw_exception = traceback.format_exc()
            record_tsm(tsm_event)
            raise

    @abc.abstractmethod
    def _schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        raise NotImplementedError()

    @abc.abstractmethod
    def _status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _wait(self, app_handle: AppHandle) -> Optional[AppStatus]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _list(self) -> Dict[AppHandle, Application]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _stop(self, app_handle: AppHandle) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def _describe(self, app_handle: AppHandle) -> Optional[Application]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterable:
        raise NotImplementedError()

    def _generate_tsm_event(
        self,
        api: str,
        scheduler: str,
        app_id: Optional[str] = None,
        runcfg: Optional[str] = None,
    ) -> TsmEvent:
        return TsmEvent(
            session=self.name(),
            scheduler=scheduler,
            api=api,
            unix_user=getpass.getuser(),
            source_hostname=socket.getfqdn(socket.gethostname()),
            app_id=app_id,
            runcfg=runcfg,
        )


class StandaloneSession(LoggingSession):
    def __init__(
        self,
        name: str,
        schedulers: Dict[SchedulerBackend, Scheduler],
        wait_interval: int = 10,
    ):
        if "default" not in schedulers:
            raise ValueError(
                f"A default scheduler is required. Provided schedulers: {schedulers.keys()}"
            )

        super().__init__(name)
        self._schedulers = schedulers
        self._wait_interval = wait_interval
        # TODO T72035686 implement an LRU cache (see local_scheduler.py) and use it here and also in local_scheduler
        self._apps: Dict[AppHandle, Application] = {}

    def _scheduler(self, scheduler: SchedulerBackend) -> Scheduler:
        sched = self._schedulers.get(scheduler)
        if not sched:
            raise KeyError(
                f"Undefined scheduler backend: {scheduler}. Use one of: {self._schedulers.keys()}"
            )
        return sched

    def _scheduler_app_id(
        self, app_handle: AppHandle, check_session: bool = True
    ) -> Tuple[Scheduler, str, str]:
        """
        Returns the scheduler and app_id from the app_handle.
        Set ``check_session`` to validate that the session name in the app handle
        is the same as this session.

        Raises:
            ValueError - if ``check_session=True`` and the session in the app handle
                         does not match this session's name
            KeyError - if no such scheduler backend exists
        """

        scheduler_backend, session_name, app_id = parse_app_handle(app_handle)
        if check_session and self._name != session_name:
            raise SessionMismatchException(app_handle, self._name)
        scheduler = self._scheduler(scheduler_backend)
        return scheduler, scheduler_backend, app_id

    def _schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        scheduler_backend = dryrun_info._scheduler
        sched = self._scheduler(scheduler_backend)
        app_id = sched.schedule(dryrun_info)
        app_handle = make_app_handle(scheduler_backend, self._name, app_id)
        self._apps[app_handle] = dryrun_info._app
        return app_handle

    def _dryrun(
        self,
        app: Application,
        scheduler: SchedulerBackend,
        cfg: RunConfig,
    ) -> AppDryRunInfo:
        sched = self._scheduler(scheduler)
        sched._validate(app, scheduler)
        return sched.submit_dryrun(app, cfg)

    def run_opts(self) -> Dict[str, runopts]:
        return {
            scheduler_backend: scheduler.run_opts()
            for scheduler_backend, scheduler in self._schedulers.items()
        }

    def scheduler_backends(self) -> List[SchedulerBackend]:
        return list(self._schedulers.keys())

    def _status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        # allow status checks of apps from other sessions
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        desc = scheduler.describe(app_id)
        if not desc:
            # app does not exist on the scheduler
            # remove it from apps cache if it exists
            # effectively removes this app from the list() API
            self._apps.pop(app_handle, None)
            return None

        app_status = AppStatus(
            desc.state,
            desc.num_restarts,
            desc.msg,
            desc.structured_error_msg,
            replicas=desc.replicas,
        )
        if app_status:
            app_status.ui_url = desc.ui_url
        return app_status

    def _wait(self, app_handle: AppHandle) -> Optional[AppStatus]:
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        while True:
            app_status = self.status(app_handle)

            if not app_status:
                return None
            if app_status.is_terminal():
                return app_status
            else:
                time.sleep(self._wait_interval)

    def _list(self) -> Dict[AppHandle, Application]:
        # opportunistically query for each app's status to update the app
        # copy the keys (app ids) since status(app_id) mutates self._apps
        app_ids = list(self._apps.keys())
        for app_id in app_ids:
            self.status(app_id)
        return self._apps

    def _stop(self, app_handle: AppHandle) -> None:
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(app_handle)
        status = self.status(app_handle)
        if status is None or status.is_terminal():
            return  # do nothing; app does not exist or has already finished
        else:
            scheduler.cancel(app_id)

    def _describe(self, app_handle: AppHandle) -> Optional[Application]:
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )

        # if the app is in the apps list, then short circuit everything and return it
        app = self._apps.get(app_handle, None)
        if app:
            return app

        desc = scheduler.describe(app_id)
        if not desc:
            return None
        else:
            return Application(name=app_id).of(*desc.roles)

    def _log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterable:
        if not self.status(app_handle):
            raise UnknownAppException(app_handle)
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )

        log_iter = scheduler.log_iter(app_id, role_name, k, regex, since, until)
        return log_iter
