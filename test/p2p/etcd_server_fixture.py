#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import closing

import etcd


log = logging.getLogger(__name__)


def _get_etcd_bin_path():
    """
    Returns the absolute path to the etcd server binary
    TORCHELASTIC_ETCD_BINARY_PATH env var is first inspected, if not set
    falls back to looking at bin/etcd relative to the current file
    """
    root = os.path.dirname(__file__)
    default_etcd_bin = os.path.join(root, "bin/etcd")
    etcd_bin = os.environ.get("TORCHELASTIC_ETCD_BINARY_PATH", default_etcd_bin)
    log.info(f"Using etcd binary in: {etcd_bin}")
    return etcd_bin


def _get_socket_with_port():
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
            log.info("Socket creation attempt failed: " + e)

    raise RuntimeError("Failed to create a socket")


class EtcdServerFixture:
    """
    Starts and stops a standalone etcd server tied to the
    lifetime of this object. Assumes etcd server version >= 3.4.3
    """

    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="torchelastic_test_")
        log.info(f"Created tmpdir {self.tmpdir}")
        self.datadir = os.path.join(self.tmpdir, "data")
        self.bindir = os.path.join(self.tmpdir, "bin")
        os.mkdir(self.datadir)
        os.mkdir(self.bindir)

        self.file, self.out_file_path = tempfile.mkstemp(
            dir=self.tmpdir, prefix="etcd_server_", suffix="out"
        )
        self._subprocess = None
        self._client_port = None
        self._peer_port = None

    def start(self, timeout=60):
        """
        starts the server and blocks for timeout seconds until the
        server is ready.
        """

        # etcd opens two ports: one for clients, and another for peers
        client_bind_sock = _get_socket_with_port()
        peer_bind_sock = _get_socket_with_port()
        with closing(client_bind_sock), closing(peer_bind_sock):
            self._client_port = client_bind_sock.getsockname()[1]
            self._peer_port = peer_bind_sock.getsockname()[1]

            # FIXME: this has a race condition here. Explore a more reliable
            # way to pass a listening port to a separate process.
            client_bind_sock.close()
            peer_bind_sock.close()
            self._subprocess = subprocess.Popen(
                self._get_etcd_cmd(),
                stdin=subprocess.PIPE,
                stdout=self.file,
                stderr=self.file,
                close_fds=True,
            )

            log.info(
                f"Starting etcd server, tmp dir: {self.tmpdir} "
                f"host: {self.get_host()}, port: {self.get_port()}"
            )

            self._client = etcd.Client(
                host=self.get_host(),
                port=self.get_port(),
                version_prefix="/v2",
                read_timeout=timeout,
            )

            # ask etcd server for version until server is ready
            version = self._get_version(timeout)
            log.info(f"etcd server v{version} is running")

    def _get_etcd_cmd(self):
        # tested on etcd server v3.4.3

        client_bind_url = f"http://{self.get_host()}:{self.get_port()}"
        peer_bind_url = f"http://{self.get_host()}:{self.get_peer_port()}"

        etcd_server_cmd = [
            _get_etcd_bin_path(),
            "--enable-v2",
            "--data-dir",
            self.datadir,
            "--listen-client-urls",
            client_bind_url,
            "--advertise-client-urls",
            client_bind_url,
            "--listen-peer-urls",
            peer_bind_url,
        ]
        log.info(f"etcd server cmd: {etcd_server_cmd}")
        return shlex.split(" ".join(etcd_server_cmd))

    def _get_version(self, timeout):
        max_time = time.time() + timeout

        while time.time() < max_time:
            try:
                return self._client.version
            except Exception:
                time.sleep(1)
        raise RuntimeError("Timed out waiting for etcd server to be ready!")

    def get_etcd_client(self):
        return self._client

    def get_host(self):
        # only allow local connections (from clients)
        return "localhost"

    def get_port(self):
        # port which clients connect to
        return self._client_port

    def get_peer_port(self):
        # port which peers (i.e. other etcd server instances) connect to
        return self._peer_port

    def stop(self):
        self._client_port = None
        self._peer_port = None

        # always check whether resource (process, file, fd) exists
        # before kill/rm so that this method is idempotent

        # stop etcd server
        if self._subprocess and self._subprocess.poll() is None:
            self._subprocess.terminate()
            self._subprocess.wait()

        shutil.rmtree(self.tmpdir, ignore_errors=True)
        log.info(f"Stopped etcd server and removed tmp data dir: {self.tmpdir}")

    def __del__(self):
        self.stop()
