#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import inspect
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import classy_vision
import torch
from classy_vision.generic.opts import check_generic_args, get_parser
from classy_vision.generic.registry_utils import import_all_packages_from_directory
from classy_vision.generic.util import load_checkpoint, load_json
from classy_vision.hooks import (
    CheckpointHook,
    LossLrMeterLoggingHook,
    ModelComplexityHook,
    ProfilerHook,
    TimeMetricsHook,
)
from classy_vision.tasks import FineTuningTask, build_task
from classy_vision.trainer.elastic_trainer import ElasticTrainer
from torch.distributed import Backend
from torchelastic.p2p import CoordinatorP2P
from torchvision import set_video_backend


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
)


# local_rank == host local rank assigned and passed by torch.multiprocessing
def main(local_rank, c10d_backend, rdzv_init_url, max_world_size, classy_args):
    torch.manual_seed(0)
    set_video_backend(classy_args.video_backend)

    # Loads config, sets up task
    config = load_json(classy_args.config_file)

    task = build_task(config)

    # Load checkpoint, if available
    checkpoint = load_checkpoint(classy_args.checkpoint_folder)
    task.set_checkpoint(checkpoint)

    pretrained_checkpoint = load_checkpoint(classy_args.pretrained_checkpoint_folder)
    if pretrained_checkpoint is not None:
        assert isinstance(
            task, FineTuningTask
        ), "Can only use a pretrained checkpoint for fine tuning tasks"
        task.set_pretrained_checkpoint(pretrained_checkpoint)

    hooks = [
        LossLrMeterLoggingHook(classy_args.log_freq),
        ModelComplexityHook(),
        TimeMetricsHook(),
    ]

    if classy_args.checkpoint_folder != "":
        args_dict = vars(classy_args)
        args_dict["config"] = config
        hooks.append(
            CheckpointHook(
                classy_args.checkpoint_folder,
                args_dict,
                checkpoint_period=classy_args.checkpoint_period,
            )
        )
    if classy_args.profiler:
        hooks.append(ProfilerHook())

    task.set_hooks(hooks)

    assert c10d_backend == Backend.NCCL or c10d_backend == Backend.GLOO
    if c10d_backend == torch.distributed.Backend.NCCL:
        # needed to enable NCCL error handling
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

    coordinator = CoordinatorP2P(
        c10d_backend=c10d_backend,
        init_method=rdzv_init_url,
        max_num_trainers=max_world_size,
        process_group_timeout=60000,
    )
    trainer = ElasticTrainer(
        use_gpu=classy_args.device == "gpu",
        num_dataloader_workers=classy_args.num_workers,
        local_rank=local_rank,
        elastic_coordinator=coordinator,
        input_args={},
    )
    trainer.train(task)


def parse_classy_args():
    """
    parses default classy args from sys.argv adding some nice-to-have
    decorations (e.g. automatically set --device depending on the host type)
    """
    parser = get_parser()
    args = parser.parse_args()

    args.config_file = to_abs_path(args.config_file)
    args.device = "gpu" if torch.cuda.is_available() else "cpu"
    check_generic_args(args)
    return args


# TODO we may want to upstream this to classy_vision utils
def to_abs_path(config_path_url):
    """
    Returns the absolute file path to the classy config file

    Get config relative to classy's module
    to_abs_path("classy-vision://config/resnet_50.json")
    -- or --

    Get config relative to this script
    to_abs_path("my_config_dir/resnet_50.json")
    -- or --

    Get config from absolute path
    to_abs_path("/absolute/config/dir/path/resnet_50.json")
    """
    config_url = urlparse(config_path_url)
    if config_url.scheme == "classy-vision":
        # read relative to classy_vision module
        classy_path = Path(inspect.getfile(classy_vision)).parent
        classy_config_file = os.path.join(
            classy_path, f"{config_url.netloc}{config_url.path}"
        )
    else:
        # read relative to script if not absolute path
        if os.path.isabs(config_url.path):
            classy_config_file = config_url.path
        else:
            classy_config_file = os.path.join(
                os.path.dirname(__file__), config_url.path
            )
    return classy_config_file


def default_local_world_size():
    """
    If CUDA is available, returns the number of GPU devices on the host.
    Otherwise returns 1.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


if __name__ == "__main__":
    # num_nodes == number of hosts participating on this job
    # assumes homogeneous hosts
    # local_world_size = number of workers to run per node
    # world_size = total number of workers
    num_nodes = os.environ.get("SIZE", 1)
    min_num_nodes = os.environ.get("MIN_SIZE", num_nodes)
    max_num_nodes = os.environ.get("MAX_SIZE", num_nodes)

    local_world_size = default_local_world_size()
    min_world_size = local_world_size * min_num_nodes
    max_world_size = local_world_size * max_num_nodes

    if torch.cuda.is_available():
        if not local_world_size:
            num_gpus = torch.cuda.device_count()
            log.info(f"Found {num_gpus} gpus on this host")
            local_world_size = num_gpus
    else:
        if not local_world_size:
            local_world_size = 1

    world_size = local_world_size * num_nodes
    log.info(f"Running {local_world_size}/{world_size} workers on this host")

    rdzv_endpoint = os.environ.get("RDZV_ENDPOINT", "localhost:2379")
    job_id = os.environ.get("JOB_ID", "torchelastic_classy_vision_example")
    rdzv_init_method = (
        f"etcd://{rdzv_endpoint}/{job_id}"
        f"?min_workers={min_world_size}"
        f"&max_workers={max_world_size}"
        f"&last_call_timeout=5"
    )
    log.info(f"rdzv init method={rdzv_init_method}")

    c10d_backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", Backend.GLOO).lower()

    file_root = Path(__file__).parent
    import_all_packages_from_directory(file_root)

    if local_world_size == 1:
        local_rank = 0
        main(
            local_rank,
            c10d_backend,
            rdzv_init_method,
            max_world_size,
            parse_classy_args(),
        )
    else:
        torch.multiprocessing.spawn(
            fn=main,
            args=(c10d_backend, rdzv_init_method, max_world_size, parse_classy_args()),
            nprocs=local_world_size,
            join=True,
        )
