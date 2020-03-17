#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from p2p.elastic_trainer_test_base import ElasticTrainerTestBase
from p2p.etcd_server_fixture import EtcdServerFixture


class EtcdElasticTrainerTest(ElasticTrainerTestBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServerFixture()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def get_rdzv_url(
        self, run_id, min_size, max_size, timeout=300, min_node_timeout=None
    ):
        host = self._etcd_server.get_host()
        port = self._etcd_server.get_port()
        # Note: last_call_timeout can potentially be a source of test flakiness
        # particularly in case of stress tests.
        return (
            f"etcd://{host}:{port}/{run_id}"
            f"?min_workers={min_size}&max_workers={max_size}"
            f"&last_call_timeout=5"
            f"&timeout={timeout}"
        )

    def test_etcd_server(self):
        # sanity check to ensure etcd server fixture works correctly
        server_version = self._etcd_server.get_etcd_client().version
        self.assertIsNotNone(server_version)
