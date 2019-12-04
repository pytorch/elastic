#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest


class VersionTest(unittest.TestCase):
    def test_can_get_version(self):
        import torchelastic

        self.assertIsNotNone(torchelastic.__version__)
