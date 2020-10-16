#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="test script")

    parser.add_argument(
        "--fail",
        default=False,
        action="store_true",
        help="forces the script to throw a RuntimeError",
    )
    parser.add_argument(
        "--wait", default=0, type=int, help="wait time in seconds befor start executing"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    time.sleep(args.wait)
    if args.fail:
        raise RuntimeError("raising exception since --fail flag was set")
    else:
        print("Success")


if __name__ == "__main__":
    main()
