#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Dict, Optional

from torchelastic.tsm.driver.api import (
    Application,
    AppNotReRunnableException,
    AppStatus,
    RunMode,
    Scheduler,
    Session,
    UnknownAppException,
)


class StandaloneSession(Session):
    def __init__(self, name: str, scheduler: Scheduler, wait_interval: int = 10):
        super().__init__(name)
        self._scheduler = scheduler
        self._wait_interval = wait_interval

        # TODO T72035686 implement an LRU cache (see local_scheduler.py) and use it here and also in local_scheduler
        # app_id -> Application
        self._apps: Dict[str, Application] = {}

    def run(self, app: Application, mode: RunMode = RunMode.HEADLESS) -> str:
        if app.is_attached:
            raise AppNotReRunnableException(app)
        return super().run(app, mode)

    def _run(self, app: Application, mode: RunMode = RunMode.HEADLESS) -> str:
        app_id = self._scheduler.submit(app, mode)
        self._apps[app_id] = app
        return app_id

    def status(self, app_id: str) -> Optional[AppStatus]:
        if app_id not in self._apps:
            raise UnknownAppException(app_id)

        desc = self._scheduler.describe(app_id)
        if not desc:
            # app does not exist on the scheduler; remove it from apps cache
            # effectively removes this app from the list() API
            del self._apps[app_id]
            return None

        app_status = AppStatus(desc.state, desc.num_restarts, desc.msg)
        app_status.ui_url = desc.ui_url
        return app_status

    def wait(self, app_id: str) -> Optional[AppStatus]:
        while True:
            app_status = self.status(app_id)

            if not app_status:
                return None
            if app_status.is_terminal():
                return app_status
            else:
                time.sleep(self._wait_interval)

    def list(self) -> Dict[str, Application]:
        # opportunistically query for each app's status to update the app
        # copy the keys (app ids) since status(app_id) mutates self._apps
        app_ids = list(self._apps.keys())
        for app_id in app_ids:
            self.status(app_id)
        return self._apps

    def stop(self, app_id: str) -> None:
        status = self.status(app_id)
        if status is None or status.is_terminal():
            return  # do nothing; app does not exist or has already finished
        else:
            self._scheduler.cancel(app_id)

    def attach(self, app_id: str) -> Application:
        # TODO recreate the app to the best extent; empty app for now (e.g. cannot re-run the app)
        app = Application(name=app_id)
        app.is_attached = True
        self._apps[app_id] = app
        return app
