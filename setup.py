#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from setuptools import find_packages, setup


if __name__ == "__main__":
    if sys.version_info < (3, 6):
        sys.exit("python >= 3.6 required for torchelastic")

    with open("README.md", encoding="utf8") as f:
        readme = f.read()

    with open("LICENSE") as f:
        license = f.read()

    with open("requirements.txt") as f:
        reqs = f.read()

    with open("VERSION") as f:
        version = f.read()

    setup(
        # Metadata
        name="torchelastic",
        version=version,
        author="PyTorch Elastic Devs",
        author_email="torchelastic@fb.com",
        description="PyTorch Elastic Training",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/elastic",
        license=license,
        python_requires=">=3.6",
        install_requires=reqs.strip().split("\n"),
        include_package_data=True,
        packages=find_packages(exclude=("test", "test.*")),
        test_suite="test.suites.unittests",
    )
