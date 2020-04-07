#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid

from torchelastic.rendezvous.etcd_rendezvous import _etcd_rendezvous_handler
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
        handler = _etcd_rendezvous_handler(
            f"etcd://{self._etcd_server.get_endpoint()}/{uuid.uuid4()}"
            f"?min_workers=1"
            f"&max_workers=1"
        )
        self.assertIsNotNone(handler)

    def test_parse_url(self):
        handler = _etcd_rendezvous_handler(
            f"etcd://{self._etcd_server.get_endpoint()}/{uuid.uuid4()}"
            f"?min_workers=1"
            f"&max_workers=1"
            f"&timeout=60"
            f"&last_call_timeout=30"
            f"&protocol=http"
        )

        self.assertIsNotNone(handler)
