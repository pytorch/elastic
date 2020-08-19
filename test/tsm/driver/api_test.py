#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from typing import Dict, Optional

from torchelastic.tsm.driver.api import (
    _TERMINAL_STATES,
    Application,
    AppState,
    AppStatus,
    Container,
    Resources,
    Role,
    RunMode,
    Session,
)


class ApplicationStatusTest(unittest.TestCase):
    def test_is_terminal(self):
        for s in AppState:
            is_terminal = AppStatus(state=s).is_terminal()
            if s in _TERMINAL_STATES:
                self.assertTrue(is_terminal)
            else:
                self.assertFalse(is_terminal)


class RoleBuilderTest(unittest.TestCase):
    def test_build_role(self):
        # runs: ENV_VAR_1=FOOBAR /bin/echo hello world
        container = Container(image="test_image")
        container.ports(foo=8080)
        trainer = (
            Role("trainer")
            .runs("/bin/echo", "hello", "world", ENV_VAR_1="FOOBAR")
            .on(container)
            .replicas(2)
        )

        self.assertEqual("trainer", trainer.name)
        self.assertEqual("/bin/echo", trainer.entrypoint)
        self.assertEqual({"ENV_VAR_1": "FOOBAR"}, trainer.env)
        self.assertEqual(["hello", "world"], trainer.args)
        self.assertEqual(container, trainer.container)
        self.assertEqual(2, trainer.num_replicas)


class ApplicationTest(unittest.TestCase):
    def test_application(self):
        container = Container(image="test_image")
        trainer = Role("trainer").runs("/bin/sleep", "10").on(container).replicas(2)
        app = Application(name="test_app").of(trainer)
        self.assertEqual("test_app", app.name)
        self.assertEqual(1, len(app.roles))
        self.assertEqual(trainer, app.roles[0])
        self.assertEqual(RunMode.HEADLESS, app.run_mode)

    def test_application_default(self):
        app = Application(name="test_app")
        self.assertEqual(RunMode.HEADLESS, app.run_mode)
        self.assertEqual(0, len(app.roles))
        self.assertFalse(app.is_attached)


class SessionTest(unittest.TestCase):
    class MockSession(Session):
        def __init__(self):
            super().__init__("mock session")

        def _run(self, app: Application, mode: RunMode = RunMode.HEADLESS) -> str:
            return app.name

        def status(self, app_id: str) -> Optional[AppStatus]:
            return None

        def wait(self, app_id: str) -> Optional[AppStatus]:
            return None

        def list(self) -> Dict[str, Application]:
            return {}

        def stop(self, app_id: str) -> None:
            pass

        def attach(self, app_id: str) -> Application:
            return Application(app_id)

    def test_validate_no_roles(self):
        session = self.MockSession()
        with self.assertRaises(ValueError):
            app = Application("no roles")
            session.run(app)

    def test_validate_no_container(self):
        session = self.MockSession()
        with self.assertRaises(ValueError):
            role = Role("no container").runs("echo", "hello_world")
            app = Application("no container").of(role)
            session.run(app)

    def test_validate_no_resource(self):
        session = self.MockSession()
        with self.assertRaises(ValueError):
            container = Container("no resource")
            role = Role("no resource").runs("echo", "hello_world").on(container)
            app = Application("no resource").of(role)
            session.run(app)

    def test_validate_invalid_replicas(self):
        session = self.MockSession()
        with self.assertRaises(ValueError):
            container = Container("torch").require(Resources(cpu=1, gpu=0, memMB=500))
            role = (
                Role("no container")
                .runs("echo", "hello_world")
                .on(container)
                .replicas(0)
            )
            app = Application("no container").of(role)
            session.run(app)
