#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
For each rst file, generates a corresponding rst file
that redirects http://pytorch.org/elastic/<version>/<file_name>.html
to http://pytorch.org/elastic/latest/<file_name>.html
"""

import argparse
import glob
import os
import sys

import torchelastic


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", required=True, help="directory where rst files are"
    )
    parser.add_argument("--build_dir", required=True, help="directory to drop md files")

    return parser.parse_args(args[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    build_ver = torchelastic.__version__
    source_dir = args.source_dir
    build_dir = args.build_dir
    print(f"Creating redirect files from source_dir: {source_dir} into {build_dir}")
    for rst_file in glob.glob(os.path.join(source_dir, "**/*.rst"), recursive=True):
        rst_relative_path = os.path.relpath(rst_file, source_dir)
        md_relative_path = os.path.splitext(rst_relative_path)[0] + ".md"
        html_relative_path = os.path.splitext(rst_relative_path)[0] + ".html"
        md_file = os.path.join(build_dir, md_relative_path)
        os.makedirs(os.path.dirname(md_file), exist_ok=True)

        print(f"Creating redirect md for {rst_relative_path} --> {md_file}")
        with open(md_file, "w") as f:
            f.write("---\n")
            f.write("layout: docs_redirect\n")
            f.write("title: PyTorch | Redirect\n")
            f.write(f'redirect_url: "/elastic/{build_ver}/{html_relative_path}"\n')
            f.write("---\n")
