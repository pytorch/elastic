#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import json
import os
import shutil
import tempfile
import unittest
from typing import Optional, Dict
from unittest import mock
from unittest.mock import MagicMock, patch

from torchelastic.events import TsmEvent
from torchelastic.tsm.driver.api import (
    Application,
    AppStatus,
    AppState,
    Container,
    DescribeAppResponse,
    Resource,
    AppDryRunInfo,
    Role,
    Session,
    AppHandle,
    RunConfig,
    SessionMismatchException,
    UnknownAppException,
    parse_app_handle,
)
from torchelastic.tsm.driver.local_scheduler import (
    LocalDirectoryImageFetcher,
    LocalScheduler,
)
from torchelastic.tsm.driver.standalone_session import StandaloneSession, LoggingSession

from .test_util import write_shell_script


class resource:
    SMALL = Resource(cpu=1, gpu=0, memMB=1024)
    MEDIUM = Resource(cpu=4, gpu=0, memMB=(4 * 1024))
    LARGE = Resource(cpu=16, gpu=0, memMB=(16 * 1024))


SESSION_NAME = "test_session"


class DummySession(LoggingSession):
    def _dryrun(self, app, scheduler, cfg):
        return None

    def scheduler_backends(self):
        return []

    def _schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        return "default://test_session/test_app"

    def _status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        return None

    def _wait(self, app_handle: AppHandle) -> Optional[AppStatus]:
        return None

    def _list(self) -> Dict[AppHandle, Application]:
        return {}

    def _stop(self, app_handle: AppHandle) -> None:
        pass

    def _describe(self, app_handle: AppHandle) -> Optional[Application]:
        return None

    def _log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime.datetime] = None,
        until: Optional[datetime.datetime] = None,
    ):
        return iter(["test_log"])


@patch("torchelastic.tsm.driver.standalone_session.record_tsm")
class LoggingSessionTest(unittest.TestCase):
    def assert_tsm_event(self, expected: TsmEvent, actual: TsmEvent):
        self.assertEqual(expected.session, actual.session)
        self.assertEqual(expected.app_id, actual.app_id)
        self.assertEqual(expected.api, actual.api)
        self.assertEqual(expected.source_hostname, actual.source_hostname)
        self.assertEqual(expected.unix_user, actual.unix_user)

    def test_status_success(self, record_tsm_mock):
        session = DummySession("test_session")
        session.status("default://test_session/test_app")
        actual_tsm_event = record_tsm_mock.call_args[0][0]  # first arg
        self.assert_tsm_event(
            session._generate_tsm_event("status", "default", "test_app"),
            actual_tsm_event,
        )

    def test_status_fail(self, record_tsm_mock):
        session = DummySession("test_session")
        with self.assertRaises(RuntimeError):
            with patch.object(session, "_status") as status_mock:
                status_mock.side_effect = RuntimeError("test error")
                session.status("default://test_session/test_app")
        record_tsm_mock.assert_called()

    def test_wait_fail(self, record_tsm_mock):
        session = DummySession("test_session")
        with self.assertRaises(RuntimeError):
            with patch.object(session, "_wait") as status_mock:
                status_mock.side_effect = RuntimeError("test error")
                session.wait("default://test_session/test_app")
        record_tsm_mock.assert_called()

    def test_describe_fail(self, record_tsm_mock):
        session = DummySession("test_session")
        with self.assertRaises(RuntimeError):
            with patch.object(session, "_describe") as status_mock:
                status_mock.side_effect = RuntimeError("test error")
                session.describe("default://test_session/test_app")
        record_tsm_mock.assert_called()

    def test_list_fail(self, record_tsm_mock):
        session = DummySession("test_session")
        with self.assertRaises(RuntimeError):
            with patch.object(session, "_list") as status_mock:
                status_mock.side_effect = RuntimeError("test error")
                session.list()
        record_tsm_mock.assert_called()

    def test_schedule_fail(self, record_tsm_mock):
        app_info = AppDryRunInfo("test", lambda x: "test")
        app_info._scheduler = "default"
        cfg = RunConfig({"image_fetcher": "dir"})
        app_info._cfg = cfg
        session = DummySession("test_session")
        with self.assertRaises(RuntimeError):
            with patch.object(session, "_schedule") as schedule_mock:
                schedule_mock.side_effect = RuntimeError("test error")
                session.schedule(app_info)
        record_tsm_mock.assert_called()

    def test_schedule_success(self, record_tsm_mock):
        app_info = AppDryRunInfo("test", lambda x: "test")
        app_info._scheduler = "default"
        cfg = RunConfig({"image_fetcher": "dir"})
        app_info._cfg = cfg
        session = DummySession("test_session")
        app_handle = session.schedule(app_info)
        actual_tsm_event = record_tsm_mock.call_args[0][0]  # first arg
        _, _, app_id = parse_app_handle(app_handle)
        self.assert_tsm_event(
            session._generate_tsm_event(
                "schedule", "default", app_id, runcfg=json.dumps(cfg.cfgs)
            ),
            actual_tsm_event,
        )


@patch("torchelastic.tsm.driver.standalone_session.record_tsm")
class StandaloneSessionTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp("StandaloneSessionTest")

        write_shell_script(self.test_dir, "touch.sh", ["touch $1"])
        write_shell_script(self.test_dir, "fail.sh", ["exit 1"])
        write_shell_script(self.test_dir, "sleep.sh", ["sleep $1"])

        self.scheduler = LocalScheduler(SESSION_NAME)
        self.cfg = RunConfig({"image_fetcher": "dir"})

        # resource ignored for local scheduler; adding as an example
        self.test_container = Container(image=self.test_dir).require(resource.SMALL)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run(self, _):
        test_file = os.path.join(self.test_dir, "test_file")
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}, wait_interval=1
        )
        self.assertEqual(1, len(session.scheduler_backends()))

        role = Role(name="touch").runs("touch.sh", test_file).on(self.test_container)
        app = Application("name").of(role)

        app_handle = session.run(app, cfg=self.cfg)
        self.assertEqual(AppState.SUCCEEDED, session.wait(app_handle).state)

    def test_dryrun(self, _):
        scheduler_mock = MagicMock()
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": scheduler_mock}, wait_interval=1
        )
        role = Role(name="touch").runs("echo", "hello world").on(self.test_container)
        app = Application("name").of(role)
        session.dryrun(app, "default", cfg=self.cfg)
        scheduler_mock.submit_dryrun.assert_called_once_with(app, self.cfg)
        scheduler_mock._validate.assert_called_once()

    def test_describe(self, _):
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}
        )
        role = Role(name="sleep").runs("sleep.sh", "60").on(self.test_container)
        app = Application("sleeper").of(role)

        app_handle = session.run(app, cfg=self.cfg)
        self.assertEqual(app, session.describe(app_handle))
        # unknown app should return None
        self.assertIsNone(session.describe("default://session1/unknown_app"))

    def test_list(self, _):
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}, wait_interval=1
        )
        role = Role(name="touch").runs("sleep.sh", "1").on(self.test_container)
        app = Application("sleeper").of(role)

        num_apps = 4

        for _ in range(num_apps):
            # since this test validates the list() API,
            # we do not wait for the apps to finish so run the apps
            # in managed mode so that the local scheduler reaps the apps on exit
            session.run(app)

        apps = session.list()
        self.assertEqual(num_apps, len(apps))

    def test_evict_non_existent_app(self, _):
        # tests that apps previously run with this session that are finished and eventually
        # removed by the scheduler also get removed from the session after a status() API has been
        # called on the app

        scheduler = LocalScheduler(session_name=SESSION_NAME, cache_size=1)
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": scheduler}, wait_interval=1
        )
        test_file = os.path.join(self.test_dir, "test_file")
        role = Role(name="touch").runs("touch.sh", test_file).on(self.test_container)
        app = Application("touch_test_file").of(role)

        # local scheduler was setup with a cache size of 1
        # run the same app twice (the first will be removed from the scheduler's cache)
        # then validate that the first one will drop from the session's app cache as well
        app_id1 = session.run(app, cfg=self.cfg)
        session.wait(app_id1)

        app_id2 = session.run(app, cfg=self.cfg)
        session.wait(app_id2)

        apps = session.list()

        self.assertEqual(1, len(apps))
        self.assertFalse(app_id1 in apps)
        self.assertTrue(app_id2 in apps)

    def test_status(self, _):
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}, wait_interval=1
        )
        role = Role(name="sleep").runs("sleep.sh", "60").on(self.test_container)
        app = Application("sleeper").of(role)
        app_handle = session.run(app, cfg=self.cfg)
        self.assertEqual(AppState.RUNNING, session.status(app_handle).state)
        session.stop(app_handle)
        self.assertEqual(AppState.CANCELLED, session.status(app_handle).state)

    def test_status_unknown_app(self, _):
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}, wait_interval=1
        )
        self.assertIsNone(session.status("default://test_session/unknown_app_id"))

    @patch("json.dumps")
    def test_status_ui_url(self, json_dumps_mock, _):
        app_id = "test_app"
        json_dumps_mock.return_value = "{}"
        mock_scheduler = MagicMock()
        resp = DescribeAppResponse()
        resp.ui_url = "https://foobar"
        mock_scheduler.submit.return_value = app_id
        mock_scheduler.describe.return_value = resp

        session = StandaloneSession(
            name="test_ui_url_session", schedulers={"default": mock_scheduler}
        )
        role = Role("ignored").runs("/bin/echo").on(self.test_container)
        app_handle = session.run(Application(app_id).of(role))
        status = session.status(app_handle)
        self.assertEquals(resp.ui_url, status.ui_url)

    @patch("json.dumps")
    def test_status_structured_msg(self, json_dumps_mock, _):
        app_id = "test_app"
        json_dumps_mock.return_value = "{}"
        mock_scheduler = MagicMock()
        resp = DescribeAppResponse()
        resp.structured_error_msg = '{"message": "test error"}'
        mock_scheduler.submit.return_value = app_id
        mock_scheduler.describe.return_value = resp

        session = StandaloneSession(
            name="test_structured_msg", schedulers={"default": mock_scheduler}
        )
        role = Role("ignored").runs("/bin/echo").on(self.test_container)
        app_handle = session.run(Application(app_id).of(role))
        status = session.status(app_handle)
        self.assertEquals(resp.structured_error_msg, status.structured_error_msg)

    def test_wait_unknown_app(self, _):
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}, wait_interval=1
        )
        self.assertIsNone(session.wait("default://test_session/unknown_app_id"))
        self.assertIsNone(session.wait("default://another_session/some_app"))

    def test_stop(self, _):
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}, wait_interval=1
        )
        self.assertIsNone(session.stop("default://test_session/unknown_app_id"))

        with self.assertRaises(SessionMismatchException):
            session.stop("default://another_session/some_app_id")

    def test_log_lines_unknown_app(self, _):
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": self.scheduler}, wait_interval=1
        )
        with self.assertRaises(UnknownAppException):
            session.log_lines("default://test_session/unknown", "trainer")

    def test_log_lines(self, _):
        app_id = "mock_app"

        scheduler_mock = MagicMock()
        scheduler_mock.describe.return_value = DescribeAppResponse(
            app_id, AppState.RUNNING
        )
        scheduler_mock.log_iter.return_value = iter(["hello", "world"])
        session = StandaloneSession(
            name=SESSION_NAME, schedulers={"default": scheduler_mock}, wait_interval=1
        )

        role_name = "trainer"
        replica_id = 2
        regex = "QPS.*"
        since = datetime.datetime.now()
        until = datetime.datetime.now()
        lines = list(
            session.log_lines(
                f"default://test_session/{app_id}",
                role_name,
                replica_id,
                regex,
                since,
                until,
            )
        )

        self.assertEqual(["hello", "world"], lines)
        scheduler_mock.log_iter.assert_called_once_with(
            app_id, role_name, replica_id, regex, since, until
        )

    def test_no_default_scheduler(self, _):
        with self.assertRaises(ValueError):
            StandaloneSession(name=SESSION_NAME, schedulers={"local": self.scheduler})

    @patch("json.dumps")
    def test_get_schedulers(self, json_dumps_mock, _):
        default_sched_mock = MagicMock()
        json_dumps_mock.return_value = "{}"
        local_sched_mock = MagicMock()
        schedulers = {"default": default_sched_mock, "local": local_sched_mock}
        session = StandaloneSession(name="test_session", schedulers=schedulers)

        role = Role(name="sleep").runs("sleep.sh", "60").on(self.test_container)
        app = Application("sleeper").of(role)
        cfg = RunConfig()
        session.run(app, scheduler="local", cfg=cfg)
        local_sched_mock.submit.called_once_with(app, cfg)
