#!/usr/bin/env python3


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
import time

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

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


class StandaloneSession(Session):
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
    ) -> Tuple[Scheduler, str]:
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
        return scheduler, app_id

    def run(
        self,
        app: Application,
        scheduler: SchedulerBackend = "default",
        cfg: Optional[RunConfig] = None,
    ) -> AppHandle:
        return super().run(app, scheduler, cfg)

    def _run(
        self,
        app: Application,
        scheduler: SchedulerBackend,
        cfg: RunConfig,
    ) -> str:
        sched = self._scheduler(scheduler)
        app_id = sched.submit(app, cfg)
        app_handle = make_app_handle(scheduler, self._name, app_id)
        self._apps[app_handle] = app
        return app_handle

    def _dryrun(
        self,
        app: Application,
        scheduler: SchedulerBackend,
        cfg: RunConfig,
    ) -> AppDryRunInfo:
        sched = self._scheduler(scheduler)
        return sched.submit_dryrun(app, cfg)

    def run_opts(self) -> Dict[str, runopts]:
        return {
            scheduler_backend: scheduler.run_opts()
            for scheduler_backend, scheduler in self._schedulers.items()
        }

    def scheduler_backends(self) -> List[SchedulerBackend]:
        return list(self._schedulers.keys())

    def status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        # allow status checks of apps from other sessions
        scheduler, app_id = self._scheduler_app_id(app_handle, check_session=False)
        desc = scheduler.describe(app_id)
        if not desc:
            # app does not exist on the scheduler
            # remove it from apps cache if it exists
            # effectively removes this app from the list() API
            self._apps.pop(app_handle, None)
            return None

        app_status = AppStatus(desc.state, desc.num_restarts, desc.msg)
        app_status.ui_url = desc.ui_url
        return app_status

    def wait(self, app_handle: AppHandle) -> Optional[AppStatus]:
        while True:
            app_status = self.status(app_handle)

            if not app_status:
                return None
            if app_status.is_terminal():
                return app_status
            else:
                time.sleep(self._wait_interval)

    def list(self) -> Dict[AppHandle, Application]:
        # opportunistically query for each app's status to update the app
        # copy the keys (app ids) since status(app_id) mutates self._apps
        app_ids = list(self._apps.keys())
        for app_id in app_ids:
            self.status(app_id)
        return self._apps

    def stop(self, app_handle: AppHandle) -> None:
        scheduler, app_id = self._scheduler_app_id(app_handle)
        status = self.status(app_handle)
        if status is None or status.is_terminal():
            return  # do nothing; app does not exist or has already finished
        else:
            scheduler.cancel(app_id)

    def describe(self, app_handle: AppHandle) -> Optional[Application]:
        scheduler, app_id = self._scheduler_app_id(app_handle, check_session=False)

        # if the app is in the apps list, then short circuit everything and return it
        app = self._apps.get(app_handle, None)
        if app:
            return app

        desc = scheduler.describe(app_id)
        if not desc:
            return None
        else:
            return Application(name=app_id).of(*desc.roles)

    def log_lines(
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

        scheduler, app_id = self._scheduler_app_id(app_handle)
        return scheduler.log_iter(app_id, role_name, k, regex, since, until)
