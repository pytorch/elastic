#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["LOGLEVEL"] = "INFO"

# Since logger initialized during imoprt statement
# the log level should be set first
from torch.distributed.run import main as run_main


def main(args=None) -> None:
    run_main(args)


if __name__ == "__main__":
    main()
