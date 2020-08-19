#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List


def write_shell_script(dir: str, name: str, content: List[str]) -> str:
    """
    Creates and writes a bash script in the specified dir with the given name.
    The contents of the script are taken from the ``content`` parameter
    where each item in the list is written as a line in the script.

    Example: ``write_shell_script("/tmp", "foobar", ["sleep 10", "echo hello world"])

    ::

    # creates /tmp/foobar with content below
    #! bin/bash

    sleep 10
    echo hello world

    """

    script_path = os.path.join(dir, name)
    with open(script_path, "w") as f:
        f.write("#! /bin/bash\n")
        for line in content:
            f.write(f"{line}\n")
        f.write("\n")

    os.chmod(script_path, 0o755)
    return script_path
