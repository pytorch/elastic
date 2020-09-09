#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="sleep script")

    parser.add_argument(
        "--sleep", default=600, type=int, help="number of seconds to sleep for"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
