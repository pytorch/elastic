#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import io
import logging
import os
import time
import typing

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchelastic
import torchelastic.distributed as edist
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from torchelastic.p2p.coordinator_p2p import CoordinatorP2P
from torchelastic.utils.data import CyclingIterator, ElasticDistributedSampler
from torchvision.models.resnet import BasicBlock, Bottleneck


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
)


class TrainParams(typing.NamedTuple):
    num_data_workers: int = 8
    num_epochs: int = 90
    base_learning_rate: float = 0.0125
    batch_per_device: int = 32
    benchmark_num_iter: int = 500
    benchmark_ddp_bucket_size: int = 25


def adjust_learning_rate(world_size, params, optimizer, epoch, num_iter, iter_index):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """

    # Trick: lr scales linearly with world size with warmup
    if epoch < 5:
        lr_step = (world_size - 1) * params.base_learning_rate / (5.0 * num_iter)
        lr = params.base_learning_rate + (epoch * num_iter + iter_index) * lr_step
    elif epoch < 80:
        lr = world_size * params.base_learning_rate * (0.1 ** (epoch // 30))
    else:
        lr = world_size * params.base_learning_rate * (0.1 ** 3)
    for param_group in optimizer.param_groups:
        lr_old = param_group["lr"]
        param_group["lr"] = lr
        # Trick: apply momentum correction when lr is updated
        if lr > lr_old:
            param_group["momentum"] = lr / lr_old * 0.9  # momentum
        else:
            param_group["momentum"] = 0.9  # default momentum
    return


class ImagenetState(torchelastic.State):
    """
    Client-provided State object; it is serializable and captures the entire
    state needed for executing one iteration of training
    """

    def __init__(self, model, params, dataset, num_epochs, epoch=0):
        self.model = model
        self.params = params
        self.dataset = dataset
        self.total_batch_size = params.batch_per_device

        self.num_epochs = num_epochs
        self.epoch = epoch

        self.iteration = 0
        self.data_start_index = 0
        self.model_state = {}

    def sync(self, world_size, rank):
        self._sync_state(rank)

        # re-initialize model
        self._init_model()

        # re-initialize data loader
        self._init_data_loader()

        return self

    def capture_snapshot(self):
        # need only capture mutable fields
        snapshot = {}
        snapshot["epoch"] = self.epoch
        snapshot["iteration"] = self.iteration
        snapshot["data_start_index"] = self.data_start_index
        snapshot["model_state"] = copy.deepcopy(self.model_state)
        return snapshot

    def apply_snapshot(self, snapshot):
        self.epoch = snapshot["epoch"]
        self.iteration = snapshot["iteration"]
        self.data_start_index = snapshot["data_start_index"]
        self.model_state = snapshot["model_state"]

    def _sync_state(self, rank):
        # broadcast from the max rank with the biggest start index
        max_rank, _ = edist.all_gather_return_max_long(self.data_start_index)

        # Broadcast the state from max_rank
        buffer = io.BytesIO()
        self.save(buffer)
        state_tensor = torch.ByteTensor(list(buffer.getvalue()))
        state_size = torch.LongTensor([state_tensor.size()])
        dist.broadcast(state_size, src=max_rank)

        if rank != max_rank:
            state_tensor = torch.ByteTensor([0] * state_size[0])

        dist.broadcast(state_tensor, src=max_rank)

        buffer = io.BytesIO(state_tensor.numpy().tobytes())
        self.load(buffer)

        log.info(
            f"Rank {rank}: Model state synced from rank: {max_rank}\n"
            f"\tbatch_size={self.batch_size}"
            f"\tdata_start_index={self.data_start_index}"
            f"\titeration={self.iteration}"
            f"\tepoch={self.epoch}/{self.num_epochs}"
        )

    def _init_model(self):
        local_rank = dist.get_rank() % torch.cuda.device_count()

        self.dist_model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],  # Tells DDP to work on a single GPU
            output_device=local_rank,  # Tells DDP to work on a single GPU
            broadcast_buffers=False,
            check_reduction=True,
        )

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(
            self.dist_model.parameters(),
            self.params.base_learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        if self.data_start_index > 0:
            self.dist_model.load_state_dict(self.model_state)

    def _data_iter_generator_fn(self, epoch):
        self.epoch = epoch
        sampler = ElasticDistributedSampler(
            dataset=self.dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            start_index=self.data_start_index,
        )
        sampler.set_epoch(epoch)

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.total_batch_size,
            shuffle=(sampler is None),
            num_workers=self.params.num_data_workers,
            pin_memory=True,
            sampler=sampler,
        )

        return iter(self.data_loader)

    def _init_data_loader(self):
        self.data_iter = CyclingIterator(
            n=self.num_epochs,
            generator_fn=self._data_iter_generator_fn,
            start_epoch=self.epoch,
        )


def single_trainer(
    local_rank,
    world_size,
    c10d_backend,
    rdzv_init_url,
    model_arch,
    training_params,
    input_path,
):
    """
    Single GPU trainer that will only train on the GPU specified by local_rank

    """

    log.info("Training world size: {}".format(world_size))

    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        input_path,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    model = models.__dict__[model_arch]()
    # Apply ResNet training in one hour's tricks to the model itself
    # to maintain the accuracy
    for m in model.modules():
        # Trick 1: the last BatchNorm layer in each block need to
        # be initialized as zero gamma
        if isinstance(m, BasicBlock):
            num_features = m.bn2.num_features
            m.bn2.weight = Parameter(torch.zeros(num_features))
            if isinstance(m, Bottleneck):
                num_features = m.bn3.num_features
                m.bn3.weight = Parameter(torch.zeros(num_features))
            # Trick 2: linear layers are initialized by
            # drawing weights from a zero-mean Gaussian with
            # standard deviation of 0.01. In the paper it was only
            # fc layer, but in practice we found this better for
            # accuracy.
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    model.train()

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    log.info(f"Rank [{local_rank}] running on GPU [{device}]")
    model.cuda()

    coordinator = CoordinatorP2P(
        c10d_backend=c10d_backend,
        init_method=rdzv_init_url,
        max_num_trainers=world_size,
        process_group_timeout=60000,
    )

    state = ImagenetState(
        model=model,
        params=training_params,
        dataset=train_dataset,
        num_epochs=training_params.num_epochs,
    )
    torchelastic.train(coordinator, train_step, state)


def train_step(state: ImagenetState):
    """
    The client-provided train_step(); it does one iteration of training
    """

    start = time.time()
    input, target = next(state.data_iter)

    # This is needed because the world size may change between iterations
    world_size = dist.get_world_size()
    # Adjust the learning rate based on the epoch
    adjust_learning_rate(
        world_size,
        state.params,
        state.optimizer,
        state.epoch,
        len(state.data_loader),
        state.iteration,
    )

    target = target.cuda(non_blocking=True)
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # Compute output
    output = state.dist_model(input_var)
    loss = state.criterion(output, target_var)

    # Compute gradient and do SGD step
    state.optimizer.zero_grad()
    loss.backward()
    state.optimizer.step()

    # Only log for "local master" - assumes homogeneous # gpus per node
    if dist.get_rank() % torch.cuda.device_count() == 0:
        log.info("Epoch: [{0}][{1}]\t".format(state.epoch, state.iteration))

    state.data_start_index += world_size * state.total_batch_size
    state.iteration += 1
    state.model_state = state.dist_model.state_dict()

    end = time.time()
    # each train_step processes one mini_batch
    # measuring wall-clock time on the host may not be totally accurate
    # as CUDA kernels are asynchronous, this is for illustration purposes only
    batch_per_sec = 1 / (end - start)
    return state, torchelastic.SimpleWorkerStats(batch_per_sec)


def default_local_world_size():
    """
    If CUDA is available, returns the number of GPU devices on the host.
    Otherwise returns 1.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def default_device():
    """
    gpu if this host has a GPU, otherwise cpu
    """
    return "gpu" if torch.cuda.is_available() else "cpu"


def main():
    # these parameters should typically be set by the scheduler/resource manager
    # hence read them from environment variables rather than program args
    num_nodes = os.environ.get("NUM_NODES", 1)
    rdzv_endpoint = os.environ.get("RDZV_ENDPOINT", "localhost:2379")
    job_id = os.environ.get("JOB_ID", "torchelastic_imagenet_example")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the directory containing the dataset",
    )

    parser.add_argument(
        "--local_world_size",
        type=int,
        default=default_local_world_size(),
        help="Number of workers to spawn locally",
    )

    parser.add_argument(
        "--num_data_workers", type=int, default=8, help="Number of data loader workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--model_arch", default="resnet101", help="Model architecture (see)"
    )

    parser.add_argument(
        "--c10d_backend", default="gloo", choices=["gloo", "nccl"], help="c10d backend"
    )

    args = parser.parse_args()
    training_params = TrainParams(
        num_data_workers=args.num_data_workers,
        num_epochs=args.epochs,
        base_learning_rate=0.1,
        batch_per_device=32,
        benchmark_num_iter=500,
        benchmark_ddp_bucket_size=25,
    )

    world_size = args.local_world_size * num_nodes
    rdzv_init_method = (
        f"etcd://{rdzv_endpoint}/{job_id}"
        f"?min_workers={world_size}"
        f"&max_workers={world_size}"
    )

    if args.local_world_size == 1:
        local_rank = 0
        single_trainer(
            local_rank,
            world_size,
            args.c10d_backend,
            rdzv_init_method,
            args.model_arch,
            training_params,
            args.input_path,
        )
    else:
        mp.spawn(
            fn=single_trainer,
            nprocs=args.local_world_size,
            args=(
                world_size,
                args.c10d_backend,
                rdzv_init_method,
                args.model_arch,
                training_params,
                args.input_path,
            ),
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
