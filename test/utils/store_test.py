#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torchelastic.utils.store as store_util


class TestStore:
    def get(self, key: str):
        return f"retrieved:{key}"


class StoreUtilTest(unittest.TestCase):
    def test_get_data(self):
        store = TestStore()
        data = store_util.get_all(store, "test/store", 10)
        for idx in range(0, 10):
            self.assertEqual(f"retrieved:test/store{idx}", data[idx])
