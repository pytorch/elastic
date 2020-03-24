#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torchelastic.rendezvous.parameters as parameters
from torch.distributed import register_rendezvous_handler


class MockedRdzv(object):
    pass


def get_mock_rdzv(url):
    return MockedRdzv()


register_rendezvous_handler("mocked-rdzv", get_mock_rdzv)


class RendezvousHandlerFactoryTest(unittest.TestCase):
    def test_construct_rdzv_url(self):
        params = parameters.RendezvousParameters(
            "etcd",
            "localhost:8081",
            "1234",
            1,
            4,
            "timeout=60,protocol=https,key=/etc/kubernetes/certs/client.key",
        )
        actual_url = parameters._construct_rendezvous_url(params)

        expected_url = (
            "etcd://localhost:8081/1234"
            "?min_workers=1"
            "&max_workers=4"
            "&timeout=60"
            "&protocol=https"
            "&key=/etc/kubernetes/certs/client.key"
        )

        self.assertEqual(expected_url, actual_url)

    def test_get_rdzv_url_no_conf(self):
        params = parameters.RendezvousParameters("etcd", "localhost:8081", "1234", 1, 4)

        actual_url = parameters._construct_rendezvous_url(params)

        expected_url = "etcd://localhost:8081/1234" "?min_workers=1" "&max_workers=4"

        self.assertEqual(expected_url, actual_url)

    def test_construct_rdzv(self):
        params = parameters.RendezvousParameters(
            "mocked-rdzv", "localhost:8081", "1234", 1, 4
        )

        rdzv = parameters.get_rendezvous(params)
        self.assertTrue(rdzv.__class__ is MockedRdzv)
