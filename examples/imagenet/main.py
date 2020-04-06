#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Source: `pytorch imagenet example <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_ # noqa B950

Modified and simplified to make the original pytorch example compatible with
torchelastic.distributed.launch.

Changes:

1. Removed ``rank``, ``gpu``, ``multiprocessing-distributed``, ``dist_url`` options.
   These are obsolete parameters when using ``torchelastic.distributed.launch``.

2. Removed ``seed``, ``evaluate``, ``pretrained`` options for simplicity.

3. Removed ``resume``, ``start-epoch`` options.
   Loads the most recent checkpoint by default.

4. ``batch-size`` is now per GPU (worker) batch size rather than for all GPUs.

5. Defaults ``workers`` (num data loader workers) to ``0``.

Usage

::

 >>> python -m torchelastic.distributed.launch
        --nnodes=$NUM_NODES
        --nproc_per_node=$WORKERS_PER_NODE
        --rdzv_id=$JOB_ID
        --rdzv_backend=etcd
        --rdzv_endpoint=$ETCD_HOST:$ETCD_PORT
        main.py
        --arch resnet18
        --epochs 20
        --batch-size 32
        <DATA_DIR>
"""

import argparse
import os
import shutil
import time
from typing import List, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchelastic.utils.data import ElasticDistributedSampler


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Elastic ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 32), per worker (GPU)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--dist-backend",
    default="nccl",
    choices=["nccl", "gloo"],
    type=str,
    help="distributed backend",
)
parser.add_argument(
    "--checkpoint-file",
    default="checkpoint.pth.tar",
    type=str,
    help="checkpoint file path, to load and save to",
)


def main():
    args = parser.parse_args()

    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    dist.init_process_group(backend=args.dist_backend, init_method="env://")

    model, criterion, optimizer = initialize_model(
        args.arch, args.lr, args.momentum, args.weight_decay, device_id
    )

    # resume from checkpoint if one exists
    start_epoch, best_acc1 = load_checkpoint(
        args.checkpoint_file, device_id, model, optimizer
    )
    print(f"=> start_epoch: {start_epoch}, best_acc1: {best_acc1}")

    train_loader, val_loader = initialize_data_loader(
        args.data, args.batch_size, args.workers
    )

    print_freq = args.print_freq
    for epoch in range(start_epoch, args.epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device_id, print_freq)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device_id, print_freq)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if device_id == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "best_acc1": best_acc1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.checkpoint_file,
            )


def initialize_model(
    arch: str, lr: float, momentum: float, weight_decay: float, device_id: int
):
    print(f"=> creating model: {arch}")
    model = models.__dict__[arch]()
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(model, device_ids=[device_id])
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device_id)
    optimizer = SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )
    return model, criterion, optimizer


def initialize_data_loader(
    data_dir, batch_size, num_data_workers
) -> Tuple[DataLoader, DataLoader]:
    traindir = os.path.join(data_dir, "train")
    valdir = os.path.join(data_dir, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                # pyre-fixme[16]: Module `transforms` has no attribute
                #  `RandomResizedCrop`.
                transforms.RandomResizedCrop(224),
                # pyre-fixme[16]: Module `transforms` has no attribute
                #  `RandomHorizontalFlip`.
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    train_sampler = ElasticDistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_data_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    model: DistributedDataParallel,
    optimizer,  # SGD
) -> Tuple[int, float]:
    start_epoch = 0
    best_acc1 = 0
    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint: {checkpoint_file}")
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(checkpoint_file, map_location=f"cuda:{device_id}")
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"=> loaded checkpoint: {checkpoint_file}")
    return start_epoch, best_acc1


def train(
    train_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    optimizer,  # SGD,
    epoch: int,
    device_id: int,
    print_freq: int,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(device_id, non_blocking=True)
        target = target.cuda(device_id, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)


def validate(
    val_loader: DataLoader,
    model: DistributedDataParallel,
    criterion,  # nn.CrossEntropyLoss
    device_id: int,
    print_freq: int,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if device_id is not None:
                images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def save_checkpoint(state, is_best: bool, filename: str):
    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = os.path.join(filename, ".tmp")
    torch.save(state, tmp_filename)
    os.rename(tmp_filename, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch: int, lr: float) -> None:
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    learning_rate = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
