#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from typing import Tuple

from torchelastic.rendezvous import (
    RendezvousHandler,
    RendezvousHandlerFactory,
    RendezvousParameters,
)


def create_mock_rdzv_handler(ignored: RendezvousParameters) -> RendezvousHandler:
    return MockRendezvousHandler()


class MockRendezvousHandler(RendezvousHandler):
    def next_rendezvous(
        self,
        # pyre-ignore[11]: Annotation `Store` is not defined as a type.
        # pyre-ignore[10]: Name `torch` is used but not defined.
    ) -> Tuple["torch.distributed.Store", int, int]:  # noqa F821
        raise NotImplementedError()

    def is_closed(self) -> bool:
        return False

    def set_closed(self):
        pass

    def num_nodes_waiting(self) -> int:
        return -1

    def get_run_id(self) -> str:
        return ""


class RendezvousHandlerFactoryTest(unittest.TestCase):
    def test_double_registration(self):
        factory = RendezvousHandlerFactory()
        factory.register("mock", create_mock_rdzv_handler)
        with self.assertRaises(ValueError):
            factory.register("mock", create_mock_rdzv_handler)

    def test_no_factory_method_found(self):
        factory = RendezvousHandlerFactory()
        rdzv_params = RendezvousParameters(
            backend="mock", endpoint="", run_id="foobar", min_nodes=1, max_nodes=2
        )

        with self.assertRaises(ValueError):
            factory.create_rdzv_handler(rdzv_params)

    def test_create_rdzv_handler(self):
        rdzv_params = RendezvousParameters(
            backend="mock", endpoint="", run_id="foobar", min_nodes=1, max_nodes=2
        )

        factory = RendezvousHandlerFactory()
        factory.register("mock", create_mock_rdzv_handler)
        mock_rdzv_handler = factory.create_rdzv_handler(rdzv_params)
        self.assertTrue(isinstance(mock_rdzv_handler, MockRendezvousHandler))


class RendezvousParametersTest(unittest.TestCase):
    def test_get_or_default(self):

        params = RendezvousParameters(
            backend="foobar",
            endpoint="localhost",
            run_id="1234",
            min_nodes=1,
            max_nodes=1,
            timeout1=None,
            timeout2=10,
        )
        self.assertEqual(30, params.get("timeout1", 30))
        self.assertEqual(10, params.get("timeout2", 20))
        self.assertEqual(60, params.get("timeout3", 60))

    def test_get(self):
        params = RendezvousParameters(
            backend="foobar",
            endpoint="localhost",
            run_id="1234",
            min_nodes=1,
            max_nodes=1,
            timeout1=None,
            timeout2=10,
        )

        with self.assertRaises(KeyError):
            params.get("timeout3")

        with self.assertRaises(KeyError):
            params.get("timeout1")

        self.assertEqual(10, params.get("timeout2"))
