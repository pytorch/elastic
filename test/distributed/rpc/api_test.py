#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torchelastic.distributed.rpc as rpc
from torch.distributed.rpc.backend_registry import BackendType


class TestRpc(unittest.TestCase):
    def test_init_app(self):
        rpc.init_app(
            role="trainer", backend=BackendType.PROCESS_GROUP, backend_options=None
        )
        pass

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
