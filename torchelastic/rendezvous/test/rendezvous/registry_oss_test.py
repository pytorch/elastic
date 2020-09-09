#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid

import torchelastic.rendezvous.registry as rdzv_registry_oss
from torchelastic.rendezvous import RendezvousParameters
from torchelastic.rendezvous.etcd_server import EtcdServer


class EtcdRendezvousTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def test_get_etcd_rdzv_handler(self):
        """
        Check that we can create the handler with a minimum set of
        params
        """
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=f"{uuid.uuid4()}",
            min_nodes=1,
            max_nodes=1,
        )
        etcd_rdzv = rdzv_registry_oss.get_rendezvous_handler(rdzv_params)
        self.assertIsNotNone(etcd_rdzv)
