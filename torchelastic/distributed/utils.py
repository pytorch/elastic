#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch.distributed as dist


def get_rank():
    """
    Simple wrapper for correctly getting rank in both distributed
    / non-distributed settings
    """
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
