#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import atexit
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time

import etcd


log = logging.getLogger(__name__)


def find_free_port():
    """
    Finds a free port and binds a temporary socket to it so that
    the port can be "reserved" until used.

    .. note:: the returned socket must be closed before using the port,
              otherwise a ``address already in use`` error will happen.
              The socket should be held and closed as close to the
              consumer of the port as possible since otherwise, there
              is a greater chance of race-condition where a different
              process may see the port as being free and take it.

    Returns: a socket binded to the reserved free port

    Usage::

    sock = find_free_port()
    port = sock.getsockname()[1]
    sock.close()
    use_port(port)
    """
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            print("Socket creation attempt failed: " + e)
    raise RuntimeError("Failed to create a socket")


def stop_etcd(subprocess, data_dir):
    if subprocess and subprocess.poll() is None:
        log.info(f"stopping etcd server")
        subprocess.terminate()
        subprocess.wait()

    log.info(f"deleting etcd data dir: {data_dir}")
    shutil.rmtree(data_dir, ignore_errors=True)


class EtcdServer:
    """
    .. note:: tested on etcd server v3.4.3

    Starts and stops a local standalone etcd server on a random free
    port. Useful for single node, multi-worker launches or testing,
    where a sidecar etcd server is more convenient than having to
    separately setup an etcd server.

    This class registers a termination handler to shutdown the etcd
    subprocess on exit. This termination handler is NOT a substitute for
    calling the ``stop()`` method.

    The following fallback mechanism is used to find the etcd binary:

    1. Uses env var TORCHELASTIC_ETCD_BINARY_PATH
    2. Uses ``<this file root>/bin/etcd`` if one exists
    3. Uses ``etcd`` from ``PATH``

    Usage
    ::

     server = EtcdServer("/usr/bin/etcd", 2379, "/tmp/default.etcd")
     server.start()
     client = server.get_client()
     # use client
     server.stop()

    Args:
        etcd_binary_path: path of etcd server binary (see above for fallback path)
    """

    def __init__(self):
        self._port = -1
        self._host = "localhost"

        root = os.path.dirname(__file__)
        default_etcd_bin = os.path.join(root, "bin/etcd")
        self._etcd_binary_path = os.environ.get(
            "TORCHELASTIC_ETCD_BINARY_PATH", default_etcd_bin
        )
        if not os.path.isfile(self._etcd_binary_path):
            self._etcd_binary_path = "etcd"

        self._data_dir = tempfile.mkdtemp(prefix="torchelastic_etcd_data")
        self._etcd_cmd = None
        self._etcd_proc = None

    def get_port(self) -> int:
        """
        Returns:
            the port the server is running on.
        """
        return self._port

    def get_host(self) -> str:
        """
        Returns:
            the host the server is running on.
        """
        return self._host

    def get_endpoint(self) -> str:
        """
        Returns:
            the etcd server endpoint (host:port)
        """
        return f"{self._host}:{self._port}"

    def start(self, timeout: int = 60) -> None:
        """
        Starts the server, and waits for it to be ready. When this function
        returns the sever is ready to take requests.

        Args:
            timeout: time (in seconds) to wait for the server to be ready
                before giving up.

        Raises:
            TimeoutError: if the server is not ready within the specified timeout
        """

        sock = find_free_port()
        sock_peer = find_free_port()
        self._port = sock.getsockname()[1]
        peer_port = sock_peer.getsockname()[1]

        etcd_cmd = shlex.split(
            " ".join(
                [
                    self._etcd_binary_path,
                    "--enable-v2",
                    "--data-dir",
                    self._data_dir,
                    "--listen-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--advertise-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--listen-peer-urls",
                    f"http://{self._host}:{peer_port}",
                ]
            )
        )

        log.info(f"Starting etcd server: [{etcd_cmd}]")

        sock.close()
        sock_peer.close()
        self._etcd_proc = subprocess.Popen(etcd_cmd, close_fds=True)
        atexit.register(stop_etcd, self._etcd_proc, self._data_dir)
        self._wait_for_ready(timeout)

    def get_client(self) -> etcd.Client:
        """
        Returns:
           An etcd client object that can be used to make requests to
           this server.
        """
        return etcd.Client(
            host=self._host, port=self._port, version_prefix="/v2", read_timeout=10
        )

    def _wait_for_ready(self, timeout: int = 60):
        client = etcd.Client(
            host=f"{self._host}", port=self._port, version_prefix="/v2", read_timeout=5
        )
        max_time = time.time() + timeout

        while time.time() < max_time:
            try:
                log.info(f"etcd server ready. version: {client.version}")
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError("Timed out waiting for etcd server to be ready!")

    def stop(self) -> None:
        """
        Stops the server and cleans up auto generated resources (e.g. data dir)
        """
        stop_etcd(self._etcd_proc, self._data_dir)
