#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy
import torch


def convert_to_distributed_tensor(tensor):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor, orig_device):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


def is_distributed_training_run():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def broadcast_long(value, src_rank):
    if is_distributed_training_run():
        tensor = torch.LongTensor([value])
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.broadcast(tensor, src=src_rank)
        tensor = convert_to_normal_tensor(tensor, orig_device)
        return tensor.item()
    return value


def broadcast_float_list(float_list, src_rank):
    if is_distributed_training_run():
        tensor = torch.Tensor(float_list)
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.broadcast(tensor, src=src_rank)
        tensor = convert_to_normal_tensor(tensor, orig_device)
        return tensor.tolist()
    return float_list


def broadcast_bool(value, src_rank):
    return broadcast_long(1 if value else 0, src_rank) == 1


def all_gather_return_max_long(value):
    """
    Returns the rank of the trainer that has the max input value and also
    the max value.
    """
    if is_distributed_training_run():
        world_size = torch.distributed.get_world_size()

        input, orig_device = convert_to_distributed_tensor(torch.LongTensor([value]))
        output = []
        for _ in range(world_size):
            output_tensor, _ = convert_to_distributed_tensor(torch.LongTensor([0]))
            output.append(output_tensor)

        torch.distributed.all_gather(output, input)

        for i in range(world_size):
            output[i] = convert_to_normal_tensor(output[i], orig_device)

        max_rank = max(range(world_size), key=lambda i: output[i][0])
        max_value = output[max_rank].tolist()
        return max_rank, max_value[0]

    return 0, value


def broadcast_model(src_rank, model):
    # Async/overlapped broadcast
    broadcast_state_work = []
    for param in model.parameters():
        broadcast_state_work.append(
            torch.distributed.broadcast(param.data, src=src_rank, async_op=True)
        )

    for work in broadcast_state_work:
        work.wait()


def broadcast_binary(data: numpy.ndarray, src_rank: int) -> numpy.ndarray:
    if not is_distributed_training_run():
        return data

    assert data is None or isinstance(
        data, numpy.ndarray
    ), "Expect `numpy.ndarray`, but got:{}".format(type(data))
    size = data.size if data is not None else 0

    # broadcast the length of target data
    size = broadcast_long(size, src_rank)
    tensor = None

    if data is not None and data.size > 0:
        tensor = torch.as_tensor(data)
    else:
        tensor = torch.zeros(size, dtype=torch.uint8)

    MIN_SIZE_FOR_CHUNK = 8 * 1024 * 1024
    CHUNKS = 8

    tensor, orig_device = convert_to_distributed_tensor(tensor)
    if size >= MIN_SIZE_FOR_CHUNK:
        # Async/overlapped broadcast
        # split tensor to 8 chunks
        chunks = torch.chunk(tensor, CHUNKS)
        broadcast_state_work = []
        # Async/overlapped broadcast
        for chunk in chunks:
            broadcast_state_work.append(
                torch.distributed.broadcast(chunk, src=src_rank, async_op=True)
            )

        logging.info("Broadcasting big binary, size: {}.".format(size))

        for work in broadcast_state_work:
            work.wait()

        logging.info("Finished broadcasting.")
        # concat chunks
        tensor = torch.cat(chunks)
    else:
        torch.distributed.broadcast(tensor, src=src_rank)

    tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor.numpy()
