#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from os.path import join
from typing import Optional
from unittest import mock
from unittest.mock import patch

from torchelastic.tsm.driver.api import (
    Application,
    AppState,
    Container,
    DescribeAppResponse,
    Role,
    RunConfig,
    NULL_RESOURCE,
    is_terminal,
    macros,
)
from torchelastic.tsm.driver.local_scheduler import (
    LocalDirectoryImageFetcher,
    LocalScheduler,
    make_unique,
)

from .test_util import write_shell_script


LOCAL_DIR_IMAGE_FETCHER_FETCH = (
    "torchelastic.tsm.driver.local_scheduler.LocalDirectoryImageFetcher.fetch"
)


class LocalDirImageFetcherTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="LocalDirImageFetcherTest")
        self.test_dir_name = os.path.basename(self.test_dir)
        self.maxDiff = None  # get full diff on assertion error

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fetch_abs_path(self):
        fetcher = LocalDirectoryImageFetcher()
        self.assertEqual(self.test_dir, fetcher.fetch(self.test_dir))

    def test_fetch_relative_path_should_throw(self):
        fetcher = LocalDirectoryImageFetcher()
        with self.assertRaises(ValueError):
            fetcher.fetch(self.test_dir_name)

    def test_fetch_does_not_exist_should_throw(self):
        non_existent_dir = join(self.test_dir, "non_existent_dir")
        fetcher = LocalDirectoryImageFetcher()
        with self.assertRaises(ValueError):
            fetcher.fetch(non_existent_dir)


LOCAL_SCHEDULER_MAKE_UNIQUE = "torchelastic.tsm.driver.local_scheduler.make_unique"

ERR_FILE_ENV = "TORCHELASTIC_ERROR_FILE"


class LocalSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")
        write_shell_script(self.test_dir, "touch.sh", ["touch $1"])
        write_shell_script(self.test_dir, "fail.sh", ["exit 1"])
        write_shell_script(self.test_dir, "sleep.sh", ["sleep $1"])
        write_shell_script(self.test_dir, "echo_stdout.sh", ["echo $1"])
        write_shell_script(self.test_dir, "echo_stderr.sh", ["echo $1 1>&2"])
        write_shell_script(
            self.test_dir,
            "echo_range.sh",
            ["for i in $(seq 0 $1); do echo $i 1>&2; sleep $2; done"],
        )
        write_shell_script(self.test_dir, "echo_env_foo.sh", ["echo $FOO 1>&2"])
        self.scheduler = LocalScheduler(session_name="test_session")
        self.test_container = Container(image=self.test_dir)

    def wait(
        self,
        app_id: str,
        scheduler: Optional[LocalScheduler] = None,
        timeout: float = 30,
    ) -> Optional[DescribeAppResponse]:
        """
        Waits for the app to finish or raise TimeoutError upon timeout (in seconds).
        If no timeout is specified waits indefinitely.

        Returns:
            The last return value from ``describe()``
        """
        scheduler_ = scheduler or self.scheduler

        interval = timeout / 100
        expiry = time.time() + timeout
        while expiry > time.time():
            desc = scheduler_.describe(app_id)

            if desc is None:
                return None
            elif is_terminal(desc.state):
                return desc

            time.sleep(interval)
        raise TimeoutError(f"timed out waiting for app: {app_id}")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_submit(self):
        # make sure the macro substitution works
        # touch a file called {app_id}_{replica_id} in the img_root directory (self.test_dir)
        test_file_name = f"{macros.app_id}_{macros.replica_id}"
        num_replicas = 2
        role = (
            Role("role1")
            .runs("touch.sh", join(f"{macros.img_root}", test_file_name))
            .on(self.test_container)
            .replicas(num_replicas)
        )
        app = Application(name="test_app").of(role)
        expected_app_id = make_unique(app.name)
        with patch(LOCAL_SCHEDULER_MAKE_UNIQUE, return_value=expected_app_id):
            cfg = RunConfig({"log_dir": self.test_dir})
            app_id = self.scheduler.submit(app, cfg)

        self.assertEqual(f"{expected_app_id}", app_id)
        self.assertEqual(AppState.SUCCEEDED, self.wait(app_id).state)

        for i in range(num_replicas):
            self.assertTrue(
                os.path.isfile(join(self.test_dir, f"{expected_app_id}_{i}"))
            )

        role = Role("role1").runs("fail.sh").on(self.test_container).replicas(2)
        app = Application(name="test_app").of(role)
        expected_app_id = make_unique(app.name)
        with patch(LOCAL_SCHEDULER_MAKE_UNIQUE, return_value=expected_app_id):
            app_id = self.scheduler.submit(app, cfg)

        self.assertEqual(f"{expected_app_id}", app_id)
        self.assertEqual(AppState.FAILED, self.wait(app_id).state)

    @mock.patch.dict(os.environ, {"FOO": "bar"})
    def test_submit_inherit_parent_envs(self):
        role = Role("echo_foo").runs("echo_env_foo.sh").on(self.test_container)
        app = Application(name="check_foo_env_var").of(role)
        app_id = self.scheduler.submit(app, RunConfig({"log_dir": self.test_dir}))
        for line in self.scheduler.log_iter(app_id, "echo_foo"):
            self.assertEqual("bar", line)

        desc = self.wait(app_id, self.scheduler)
        self.assertEqual(AppState.SUCCEEDED, desc.state)

    @mock.patch.dict(os.environ, {"FOO": "bar"})
    def test_submit_override_parent_env(self):
        role = (
            Role("echo_foo")
            .runs("echo_env_foo.sh", FOO="new_bar")
            .on(self.test_container)
        )
        app = Application(name="check_foo_env_var").of(role)
        app_id = self.scheduler.submit(app, RunConfig({"log_dir": self.test_dir}))
        for line in self.scheduler.log_iter(app_id, "echo_foo"):
            self.assertEqual("new_bar", line)

        desc = self.wait(app_id, self.scheduler)
        self.assertEqual(AppState.SUCCEEDED, desc.state)

    def _assert_file_content(self, filename: str, expected: str):
        with open(filename, "r") as f:
            self.assertEqual(expected, f.read())

    def test_submit_with_log_dir_stdout(self):
        num_replicas = 2

        for std_stream in ["stdout", "stderr"]:
            with self.subTest(std_stream=std_stream):
                log_dir = join(self.test_dir, f"test_{std_stream}_log")
                cfg = RunConfig({"log_dir": log_dir})

                role = (
                    Role("role1")
                    .runs(f"echo_{std_stream}.sh", "hello_world")
                    .on(self.test_container)
                    .replicas(num_replicas)
                )
                app = Application(name="test_app").of(role)

                app_id = self.scheduler.submit(app, cfg)
                self.wait(app_id)

                success_file = join(
                    log_dir, self.scheduler.session_name, app_id, "SUCCESS"
                )
                with open(success_file, "r") as f:
                    sf_json = json.load(f)
                    self.assertEqual(app_id, sf_json["app_id"])
                    self.assertEqual(
                        join(log_dir, self.scheduler.session_name, app_id),
                        sf_json["log_dir"],
                    )
                    self.assertEqual(AppState.SUCCEEDED.name, sf_json["final_state"])

                    for replica_id in range(num_replicas):
                        replica_info = sf_json["roles"]["role1"][replica_id]
                        self._assert_file_content(
                            replica_info[std_stream], "hello_world\n"
                        )

    @patch(LOCAL_DIR_IMAGE_FETCHER_FETCH, return_value="")
    def test_submit_dryrun_without_log_dir_cfg(self, img_fetcher_fetch_mock):
        master = (
            Role("master")
            .runs("master.par", "arg1", ENV_VAR_1="VAL1")
            .on(self.test_container)
        )
        trainer = (
            Role("trainer").runs("trainer.par").on(self.test_container).replicas(2)
        )

        app = Application(name="test_app").of(master, trainer)
        cfg = RunConfig()
        info = self.scheduler.submit_dryrun(app, cfg)
        # intentional print (to make sure it actually prints with no errors)
        print(info)

        request = info.request
        role_params = request.role_params
        role_log_dirs = request.role_log_dirs
        self.assertEqual(2, len(role_params))
        self.assertEqual(2, len(role_log_dirs))

        master_params = role_params["master"]
        trainer_params = role_params["trainer"]

        app_log_dir = request.log_dir

        self.assertEqual(1, len(master_params))
        self.assertEqual(2, len(trainer_params))

        for role in app.roles:
            replica_params = role_params[role.name]
            replica_log_dirs = role_log_dirs[role.name]

            for j in range(role.num_replicas):
                replica_param = replica_params[j]
                replica_log_dir = replica_log_dirs[j]

                # dryrun should NOT create any directories
                self.assertFalse(os.path.isdir(replica_log_dir))
                self.assertTrue(replica_log_dir.startswith(app_log_dir))
                self.assertEqual([role.entrypoint, *role.args], replica_param.args)
                self.assertEqual(
                    {
                        ERR_FILE_ENV: join(replica_log_dir, "error.json"),
                        **role.env,
                    },
                    replica_param.env,
                )
                self.assertIsNone(replica_param.stdout)
                self.assertIsNone(replica_param.stderr)

    @patch(LOCAL_DIR_IMAGE_FETCHER_FETCH, return_value="")
    def test_submit_dryrun_with_log_dir_cfg(self, img_fetcher_fetch_mock):
        trainer = (
            Role("trainer").runs("trainer.par").on(self.test_container).replicas(2)
        )

        app = Application(name="test_app").of(trainer)
        cfg = RunConfig({"log_dir": self.test_dir})
        info = self.scheduler.submit_dryrun(app, cfg)
        # intentional print (to make sure it actually prints with no errors)
        print(info)

        request = info.request
        role_params = request.role_params
        role_log_dirs = request.role_log_dirs
        self.assertEqual(1, len(role_params))
        self.assertEqual(1, len(role_log_dirs))

        trainer_params = role_params["trainer"]

        app_log_dir = request.log_dir

        self.assertEqual(2, len(trainer_params))

        for role in app.roles:
            replica_params = role_params[role.name]
            replica_log_dirs = role_log_dirs[role.name]

            for j in range(role.num_replicas):
                replica_param = replica_params[j]
                replica_log_dir = replica_log_dirs[j]
                # dryrun should NOT create any directories
                self.assertFalse(os.path.isdir(replica_log_dir))
                self.assertTrue(replica_log_dir.startswith(app_log_dir))
                self.assertEqual([role.entrypoint, *role.args], replica_param.args)
                self.assertEqual(
                    {
                        ERR_FILE_ENV: join(replica_log_dir, "error.json"),
                        **role.env,
                    },
                    replica_param.env,
                )
                stdout_path = join(replica_log_dir, "stdout.log")
                stderr_path = join(replica_log_dir, "stderr.log")
                self.assertEqual(stdout_path, replica_param.stdout)
                self.assertEqual(stderr_path, replica_param.stderr)

    def test_log_iterator(self):
        role = (
            Role("role1")
            .runs("echo_range.sh", "10", "0.5")
            .on(self.test_container)
            .replicas(1)
        )

        log_dir = join(self.test_dir, "log")
        cfg = RunConfig({"log_dir": log_dir})
        app = Application(name="test_app").of(role)
        app_id = self.scheduler.submit(app, cfg)

        for i, line in enumerate(self.scheduler.log_iter(app_id, "role1", k=0)):
            self.assertEqual(str(i), line)

        # since and until ignored
        for i, line in enumerate(
            self.scheduler.log_iter(
                app_id, "role1", k=0, since=datetime.now(), until=datetime.now()
            )
        ):
            self.assertEqual(str(i), line)

        for i, line in enumerate(
            self.scheduler.log_iter(app_id, "role1", k=0, regex=r"[02468]")
        ):
            self.assertEqual(str(i * 2), line)

    def test_log_iterator_no_log_dir(self):
        role = (
            Role("role1")
            .runs("echo_range.sh", "10", "0.5")
            .on(self.test_container)
            .replicas(1)
        )

        app = Application(name="test_app").of(role)

        with self.assertRaises(RuntimeError, msg="log_dir must be set to iterate logs"):
            app_id = self.scheduler.submit(app, RunConfig())
            self.scheduler.log_iter(app_id, "role1", k=0)

    def test_submit_multiple_roles(self):
        test_file1 = join(self.test_dir, "test_file_1")
        test_file2 = join(self.test_dir, "test_file_2")
        role1 = (
            Role("role1")
            .runs("touch.sh", test_file1)
            .on(self.test_container)
            .replicas(1)
        )
        role2 = (
            Role("role2")
            .runs("touch.sh", test_file2)
            .on(self.test_container)
            .replicas(1)
        )
        app = Application(name="test_app").of(role1, role2)
        cfg = RunConfig({"log_dir": self.test_dir})
        app_id = self.scheduler.submit(app, cfg)

        self.assertEqual(AppState.SUCCEEDED, self.wait(app_id).state)
        self.assertTrue(os.path.isfile(test_file1))
        self.assertTrue(os.path.isfile(test_file2))

    def test_describe(self):
        role = Role("role1").runs("sleep.sh", "2").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        cfg = RunConfig({"log_dir": self.test_dir})
        self.assertIsNone(self.scheduler.describe("test_app_0"))
        app_id = self.scheduler.submit(app, cfg)
        desc = self.scheduler.describe(app_id)
        self.assertEqual(AppState.RUNNING, desc.state)
        self.assertEqual(AppState.SUCCEEDED, self.wait(app_id).state)

    def test_cancel(self):
        role = Role("role1").runs("sleep.sh", "10").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        cfg = RunConfig({"log_dir": self.test_dir})
        app_id = self.scheduler.submit(app, cfg)
        desc = self.scheduler.describe(app_id)
        self.assertEqual(AppState.RUNNING, desc.state)
        self.scheduler.cancel(app_id)
        self.assertEqual(AppState.CANCELLED, self.scheduler.describe(app_id).state)

    def test_exists(self):
        role = Role("role1").runs("sleep.sh", "10").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        cfg = RunConfig({"log_dir": self.test_dir})
        app_id = self.scheduler.submit(app, cfg)

        self.assertTrue(self.scheduler.exists(app_id))
        self.scheduler.cancel(app_id)
        self.assertTrue(self.scheduler.exists(app_id))

    def test_invalid_cache_size(self):
        with self.assertRaises(ValueError):
            LocalScheduler(session_name="test_session", cache_size=0)

        with self.assertRaises(ValueError):
            LocalScheduler(session_name="test_session", cache_size=-1)

    def test_cache_full(self):
        scheduler = LocalScheduler(session_name="test_session", cache_size=1)

        role = Role("role1").runs("sleep.sh", "10").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        cfg = RunConfig({"log_dir": self.test_dir})
        scheduler.submit(app, cfg)
        with self.assertRaises(IndexError):
            scheduler.submit(app, cfg)

    def test_cache_evict(self):
        scheduler = LocalScheduler(session_name="test_session", cache_size=1)
        test_file1 = join(self.test_dir, "test_file_1")
        test_file2 = join(self.test_dir, "test_file_2")
        role1 = Role("role1").runs("touch.sh", test_file1).on(self.test_container)
        role2 = Role("role2").runs("touch.sh", test_file2).on(self.test_container)
        app1 = Application(name="touch_test_file1").of(role1)
        app2 = Application(name="touch_test_file2").of(role2)
        cfg = RunConfig({"log_dir": self.test_dir})

        app_id1 = scheduler.submit(app1, cfg)
        self.assertEqual(AppState.SUCCEEDED, self.wait(app_id1, scheduler).state)

        app_id2 = scheduler.submit(app2, cfg)
        self.assertEqual(AppState.SUCCEEDED, self.wait(app_id2, scheduler).state)

        # app1 should've been evicted
        self.assertIsNone(scheduler.describe(app_id1))
        self.assertIsNone(self.wait(app_id1, scheduler))

        self.assertIsNotNone(scheduler.describe(app_id2))
        self.assertIsNotNone(self.wait(app_id2, scheduler))
