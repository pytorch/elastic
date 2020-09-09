#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

from torchelastic.tsm.driver.api import Application, AppState, Container, Role, RunMode
from torchelastic.tsm.driver.local_scheduler import (
    LocalDirectoryImageFetcher,
    LocalScheduler,
)

from .test_util import write_shell_script


class LocalDirImageFetcherTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="LocalDirImageFetcherTest")
        self.test_dir_name = os.path.basename(self.test_dir)

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
        non_existent_dir = os.path.join(self.test_dir, "non_existent_dir")
        fetcher = LocalDirectoryImageFetcher()
        with self.assertRaises(ValueError):
            fetcher.fetch(non_existent_dir)


class LocalSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp("LocalSchedulerTest")
        write_shell_script(self.test_dir, "touch.sh", ["touch $1"])
        write_shell_script(self.test_dir, "fail.sh", ["exit 1"])
        write_shell_script(self.test_dir, "sleep.sh", ["sleep $1"])

        self.image_fetcher = LocalDirectoryImageFetcher()
        self.scheduler = LocalScheduler(self.image_fetcher)

        self.test_container = Container(image=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_submit(self):
        test_file = os.path.join(self.test_dir, "test_file")
        role = (
            Role("role1")
            .runs("touch.sh", test_file)
            .on(self.test_container)
            .replicas(2)
        )
        app = Application(name="test_app").of(role)

        app_id = self.scheduler.submit(app, RunMode.HEADLESS)

        self.assertEqual("test_app_0", app_id)
        self.assertEqual(AppState.SUCCEEDED, self.scheduler.wait(app_id).state)
        self.assertTrue(os.path.isfile(test_file))

        role = Role("role1").runs("fail.sh").on(self.test_container).replicas(2)
        app = Application(name="test_app").of(role)
        app_id = self.scheduler.submit(app, RunMode.HEADLESS)

        self.assertEqual("test_app_1", app_id)
        self.assertEqual(AppState.FAILED, self.scheduler.wait(app_id).state)

    def test_submit_multiple_roles(self):
        test_file1 = os.path.join(self.test_dir, "test_file_1")
        test_file2 = os.path.join(self.test_dir, "test_file_2")
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
        app_id = self.scheduler.submit(app, RunMode.HEADLESS)

        self.assertEqual(AppState.SUCCEEDED, self.scheduler.wait(app_id).state)
        self.assertTrue(os.path.isfile(test_file1))
        self.assertTrue(os.path.isfile(test_file2))

    def test_describe(self):
        role = Role("role1").runs("sleep.sh", "2").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        self.assertIsNone(self.scheduler.describe("test_app_0"))
        app_id = self.scheduler.submit(app, RunMode.HEADLESS)
        desc = self.scheduler.describe(app_id)
        self.assertEqual(AppState.RUNNING, desc.state)
        self.assertEqual(AppState.SUCCEEDED, self.scheduler.wait(app_id).state)

    def test_cancel(self):
        role = Role("role1").runs("sleep.sh", "10").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        app_id = self.scheduler.submit(app, RunMode.HEADLESS)
        desc = self.scheduler.describe(app_id)
        self.assertEqual(AppState.RUNNING, desc.state)
        self.scheduler.cancel(app_id)
        self.assertEqual(AppState.CANCELLED, self.scheduler.describe(app_id).state)

    def test_exists(self):
        role = Role("role1").runs("sleep.sh", "10").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        app_id = self.scheduler.submit(app, RunMode.HEADLESS)

        self.assertTrue(self.scheduler.exists(app_id))
        self.scheduler.cancel(app_id)
        self.assertTrue(self.scheduler.exists(app_id))

    def test_wait_timeout(self):
        role = Role("role1").runs("sleep.sh", "10").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        app_id = self.scheduler.submit(app, RunMode.MANAGED)

        with self.assertRaises(TimeoutError):
            self.scheduler.wait(app_id, timeout=1)

    def test_invalid_cache_size(self):
        with self.assertRaises(ValueError):
            LocalScheduler(self.image_fetcher, cache_size=0)

        with self.assertRaises(ValueError):
            LocalScheduler(self.image_fetcher, cache_size=-1)

    def test_cache_full(self):
        scheduler = LocalScheduler(self.image_fetcher, cache_size=1)

        role = Role("role1").runs("sleep.sh", "10").on(self.test_container).replicas(1)
        app = Application(name="test_app").of(role)
        scheduler.submit(app, RunMode.MANAGED)
        with self.assertRaises(IndexError):
            scheduler.submit(app, RunMode.MANAGED)

    def test_cache_evict(self):
        scheduler = LocalScheduler(self.image_fetcher, cache_size=1)
        test_file1 = os.path.join(self.test_dir, "test_file_1")
        test_file2 = os.path.join(self.test_dir, "test_file_2")
        role1 = Role("role1").runs("touch.sh", test_file1).on(self.test_container)
        role2 = Role("role2").runs("touch.sh", test_file2).on(self.test_container)
        app1 = Application(name="touch_test_file1").of(role1)
        app2 = Application(name="touch_test_file2").of(role2)

        app_id1 = scheduler.submit(app1, RunMode.HEADLESS)
        self.assertEqual(AppState.SUCCEEDED, scheduler.wait(app_id1).state)

        app_id2 = scheduler.submit(app2, RunMode.HEADLESS)
        self.assertEqual(AppState.SUCCEEDED, scheduler.wait(app_id2).state)

        # app1 should've been evicted
        self.assertIsNone(scheduler.describe(app_id1))
        self.assertIsNone(scheduler.wait(app_id1))

        self.assertIsNotNone(scheduler.describe(app_id2))
        self.assertIsNotNone(scheduler.wait(app_id2))
