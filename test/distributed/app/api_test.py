#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import unittest
from unittest.mock import patch

import torch.distributed as dist
import torch.distributed.rpc as rpc
import torchelastic.distributed.app as app
from test_utils import is_tsan
from torch.distributed.rpc.backend_registry import BackendType


def echo(msg):
    return msg


class TestStore:
    def __init__(self, _name="", _source_store=None):
        self.name = _name
        self.source_store = _source_store

    def get(self, key: str):
        return f"retrieved:{key}"


class TestRpc(unittest.TestCase):
    def test_init_app(self):
        app.init_app(
            role="trainer", backend=BackendType.PROCESS_GROUP, backend_options=None
        )

    @patch("torch.distributed.autograd._init")
    @patch("torch.distributed.rpc.api._init_rpc_backend")
    def test_init_rpc(self, rpc_backend_mock, autograd_mock):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        store = TestStore()
        app.init_rpc(
            name="trainer_worker",
            backend=BackendType.PROCESS_GROUP,
            backend_options=None,
            store=store,
        )
        autograd_mock.assert_called_once()
        rpc_backend_mock.assert_called_once()

    @unittest.skipIf(is_tsan(), "test incompatible with tsan")
    def test_custom_init_rpc(self):
        def init_rpc(rank, world_size, name):
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "24500"
            rendezvous_iterator = dist.rendezvous(
                "env://", rank=rank, world_size=world_size
            )
            store, _, _ = next(rendezvous_iterator)
            app.init_rpc(
                name=name,
                backend=BackendType.PROCESS_GROUP,
                backend_options=None,
                store=store,
            )

        def master(msg):
            init_rpc(0, 2, "master")
            ret = rpc.rpc_sync(to="worker", func=echo, args=(msg,))
            rpc.shutdown()
            return ret

        def worker():
            init_rpc(1, 2, "worker")
            rpc.shutdown()

        worker_proc = multiprocessing.Process(target=worker, args=())
        worker_proc.start()
        expected_msg = "test_message_on_worker"
        actual_msg = master(expected_msg)
        worker_proc.join()
        self.assertEqual(expected_msg, actual_msg)

    def test_get_worker_names(self):
        pass

    def test_get_role_info(self):
        pass

    def test_get_all_roles(self):
        pass

    def test_wait_all(self):
        pass

    def test_rpc_sync_on_role(self):
        pass

    def test_rpc_async_on_role(self):
        pass

    def test_rpc_remote_on_role(self):
        pass

    def test_init_process_group(self):
        pass
