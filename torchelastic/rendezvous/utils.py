# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Optional, Tuple


# Default overall timeout for the rendezvous.
_DEFAULT_RDZV_TIMEOUT: int = 600  # 10 minutes

# Additional waiting time after reaching the minimum number of workers
# for the case the rendezvous is elastic (min != max).
_DEFAULT_RDZV_LAST_CALL_TIMEOUT: int = 30  # 30 seconds


def _parse_hostname_and_port(
    endpoint: Optional[str], default_port: int
) -> Tuple[str, int]:
    """
    Extracts the hostname and the port number from an endpoint string that has
    the format <hostname>:<port>.

    If no hostname can be found, defaults to the loopback address 127.0.0.1.
    """
    if not endpoint:
        return ("127.0.0.1", default_port)

    hostname, *rest = endpoint.rsplit(":", 1)
    if len(rest) == 1:
        if re.match(r"^[0-9]{1,5}$", rest[0]):
            port = int(rest[0])
        else:
            port = 0
        if port <= 80 or port >= 2 ** 16:
            raise ValueError(f"The endpoint '{endpoint}' has an invalid port number.")
    else:
        port = default_port

    if not re.match(r"^[\w\.:-]+$", hostname):
        raise ValueError(f"The enpoint '{endpoint}' has an invalid hostname.")

    return hostname, port
