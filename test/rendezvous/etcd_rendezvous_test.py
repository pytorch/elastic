#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid

from torchelastic.rendezvous import RendezvousParameters
from torchelastic.rendezvous.etcd_rendezvous import create_rdzv_handler
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

    def test_parse_url_basic(self):
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
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        self.assertIsNotNone(etcd_rdzv)

    def test_parse_url(self):
        run_id = str(uuid.uuid4())
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            last_call_timeout=30,
            protocol="http",
        )

        etcd_rdzv = create_rdzv_handler(rdzv_params)

        self.assertIsNotNone(etcd_rdzv)
        self.assertEqual(run_id, etcd_rdzv.get_run_id())
