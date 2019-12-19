#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs

import logging
import os
import os.path
import socket
import unittest
from hashlib import md5

import numpy as np
import torch.distributed as dist
import torchelastic.distributed as edist
from torch.multiprocessing import Process, Queue


log = logging.getLogger(__name__)


def compute_checksum(data: np.ndarray) -> bytes:
    return md5(data).hexdigest().encode("utf-8")


def get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def init_process(rank, world_size, fn, input, q, backend="gloo", port=28502):
    """ Initialize the distributed environment. """
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        r = fn(rank, world_size, input)
        q.put(r)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_in_process_group(world_size, fn, input):
    assert not dist.is_initialized()
    processes = []
    q = Queue()
    port = get_free_tcp_port()
    log.info(f"using tcp port: {port}")
    backend = "gloo"
    for rank in range(world_size - 1):
        p = Process(
            target=init_process, args=(rank, world_size, fn, input, q, backend, port)
        )
        p.start()
        processes.append(p)

    if world_size >= 1:
        # run 1 process in current unittest process for debug purpose
        init_process(world_size - 1, world_size, fn, input, q, backend, port)

    for p in processes:
        p.join()
    return q


class TestStateUtil(unittest.TestCase):
    def run_and_get_result(self, world_size, fn, input):
        q = run_in_process_group(world_size, fn, input)
        results = []
        while not q.empty():
            results.append(q.get())
        return results

    def test_broadcast_long(self):
        def broadcast_run(rank, world_size, input):
            return edist.broadcast_long(input[rank], 1)

        results = self.run_and_get_result(3, broadcast_run, [7, 8, 9])
        self.assertListEqual(results, [8, 8, 8])

    def test_all_gather_return_max_long(self):
        def broadcast_run(rank, world_size, input):
            return edist.all_gather_return_max_long(input[rank])

        results = self.run_and_get_result(3, broadcast_run, [7, 8, 9])
        self.assertListEqual(results, [(2, 9), (2, 9), (2, 9)])

    def test_broadcast_bool(self):
        def broadcast_run(rank, world_size, input):
            return edist.broadcast_bool(input[rank], 1)

        results = self.run_and_get_result(3, broadcast_run, [1, 0, 1])
        self.assertListEqual(results, [0, 0, 0])

    def test_broadcast_float_list(self):
        def broadcast_run(rank, world_size, input):
            return edist.broadcast_float_list(input[rank], 1)

        results = self.run_and_get_result(
            3,
            broadcast_run,
            [[1.1, 2.2, 3.3, 4.4, 5.5], [6.6, 2.3, 3.2, 4.2, 5.1], [1, 6, 3, 4, 5]],
        )
        expect = [6.6, 2.3, 3.2, 4.2, 5.1]
        for r in results:
            for i in range(len(r)):
                self.assertAlmostEqual(r[i], expect[i], places=6)

    def _do_test_broadcast_binary(self, size):
        def broadcast_run(rank, world_size, input):
            data = edist.broadcast_binary(
                np.asarray(input[rank]) if input[rank] else None, 1
            )
            return compute_checksum(data)

        input1 = bytearray(np.random.randint(100, size=size))
        input2 = bytearray(np.random.randint(100, size=size))
        inputs = [input1, input2, None]

        results = self.run_and_get_result(3, broadcast_run, inputs)
        checksum = compute_checksum(inputs[1])
        self.assertListEqual(results, [checksum, checksum, checksum])

    def test_broadcast_binary(self):
        self._do_test_broadcast_binary(100)

    def test_broadcast_binary_big(self):
        self._do_test_broadcast_binary(10 * 1024 * 1024)
