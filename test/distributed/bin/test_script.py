#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="test script, parses --local_rank,"
        " used to test launcher with no --use_env flag"
    )

    # make this required so that if --local_rank is not passed
    # when --use_env is used, the script fails as it fails to
    # parse arguments
    parser.add_argument(
        "--local_rank", required=True, type=int, help="set by launch.py"
    )

    # file is used for assertions
    parser.add_argument(
        "--touch_file_dir",
        type=str,
        help="dir to touch a file with global rank as the filename",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running {sys.argv}")

    env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]

    print("Distributed env vars set by agent:")
    for env_var in env_vars:
        value = os.environ[env_var]
        print(f"{env_var} = {value}")

    file = os.path.join(args.touch_file_dir, os.environ["RANK"])
    Path(file).touch()
    print(f"Success, created {file}")


if __name__ == "__main__":
    main()
