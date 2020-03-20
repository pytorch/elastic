#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
This module provides similar functionality as `torch.distributed.launch`,
with the following additional functionalities:

1. Worker failures are handled gracefully by restarting all workers.
2. Worker `RANK` and `WORLD_SIZE` are assigned automatically.
3. Number of nodes is allowed to change between min and max sizes (elasticity).

**Usage:**

1. Fault tolerant (fixed sized number of workers, no elasticity):

::

    >>> python -m torchelastic.distributed.launch
            --nnodes=4
            --nproc_per_node=8
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            --nproc_per_node=8
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

2. Elastic (min=1, max=4):

::

    >>> python -m torchelastic.distributed.launch
            --nnodes=1:4
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            --nproc_per_node=8
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

**Definitions:**

1. `Node` - Physical instance or container.
Maps to the unit that the job manager works with.

2. `Worker` - A worker in the context of distributed training.

3. `Worker Group` - Workers with the same function (e.g. trainers)

4. `Local Worker Group` - Subset of the workers in the
worker group running on the same Node

4. `RANK` - rank of the worker within a worker group.

5. `WORLD_SIZE` - total number of workers in a worker group.

6. `LOCAL_RANK` - rank of the worker within a local worker group

7. `LOCAL_WORLD_SIZE` - size of the local worker group

8. `rdzv_id` - user defined id that uniquely identifies the worker group
for a job. This id is used by each node to join as a member of a particular
worker group.

9. `rdzv_backend` - the backend store of rendezvous (e.g. etcd). This is
typically a strongly consistent key-value store.

10. `rdzv_endpoint` - rdzv backend server endpoint in `host:port` format.

A `Node` runs `LOCAL_WORLD_SIZE` workers which comprise a `LocalWorkerGroup`.
The union of all `LocalWorkerGroups` in the nodes in the job comprise the
`WorkerGroup`.

**Deployment:**

0. Start the rdzv backend server and get the endpoint
(to be passed as `--rdzv_endpoint` to the launcher script)

1. Single-node multi-worker - start the launcher on the host to start
the agent process which creates and monitors a local worker group.

2. Multi-node multi-worker - Start the launcher with the same arguments
on all the nodes participating in training.

When using a job/cluster manager the entry point command to the multi-node
job is invoking this launcher.

**Failure Modes:**

1. Worker failure - For a training job with `n` workers, if `k < n` workers fail
all workers are stopped and restarted up to `max_restarts`.

2. Agent failure - An agent failure results in local worker group failure,
it is up to the job manager to fail the entire job (gang semantics) or attempt
to replace the node. Both behaviors are supported by the agent.

3. Node failure - Same as agent failure.

**Membership Changes:**

1. Node departure (scale-down) - agent is notified of the departure,
all existing workers are stopped, a new `Worker Group` is formed and all
workers are started with a new `RANK` and `WORLD_SIZE`.

2. Node arrival (scale-up) - the new node is admitted to the job,
all existing workers are stopped, a new `Worker Group` is formed and all
workers are started with a new `RANK` and `WORLD_SIZE`.


**Important Notices:**

1. All the items in the important notices section of `torch.distributed.launch`
apply to this module as well

2. The environment variables necessary to initialize a torch process group
are provided to you by this module, no need for you to pass `RANK` manually.
To initialize a process group in your training script, simply run

::
    >>> torch.distributed.init_process_group(backend="gloo|nccl",
                                             init_method="env://")

3. On failures or membership changes ALL surviving workers are killed
immediately. Make sure to checkpoint your progress. The frequency of
checkpoints should depend on your job's tolerance for lost work.

4. This module only supports homogeneous `LOCAL_WORLD_SIZE`. That is,
it is assumed that all nodes run the same number of local workers (per role).

5. `RANK` is NOT stable. Between restarts, the local workers on a node
can be assgined a different range of ranks than before. NEVER hard code
any assumptions about the stable-ness of ranks or some correlation between
`RANK` and `LOCAL_RANK`.

6. When using elasticity (`min_size != max_size`) DO NOT hard code
assumptions about `WORLD_SIZE` as the world size can change due as
nodes are allowed to leave and join.

7. It is recommended your script have the following structure

::
    def main():
        pre_initialize()
        load_checkpoint(checkpoint_path)
        initialize()
        start_train()

    def start_train():
        while not end_of_data:
            train_step()
            if should_checkpoint:
                save_checkpoint(checkpoint_path)
"""

import os
import subprocess
import sys
from argparse import REMAINDER, ArgumentParser

import torch.distributed as dist
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
from torchelastic.agent.server.api import WorkerSpec
from torchelastic.agent.server.local_elastic_agent import LocalElasticAgent


def parse_args(args):
    """
    Helper function parsing the command line options.
    """

    parser = ArgumentParser(description="torchelastic elastic training launcher")

    # Arguments for the launch helper
    # worker/node size related arguments
    parser.add_argument(
        "--nnodes",
        type=str,
        default="1:1",
        help="number of nodes or MIN_NODES:MAX_NODES",
    )
    parser.add_argument(
        "--nproc_per_node", type=int, default=1, help="number of workers per node"
    )

    # rendezvous related arguments
    parser.add_argument(
        "--rdzv_backend", type=str, default="etcd", help="rendezvous backend"
    )
    parser.add_argument(
        "--rdzv_endpoint",
        type=str,
        default="",
        help="rendezvous backend server host:port",
    )
    parser.add_argument("--rdzv_id", type=str, help="user defined group id")
    parser.add_argument(
        "--rdzv_conf",
        type=str,
        default="",
        help="additional rdzv configuration (conf1=v1,conf2=v2,...)",
    )

    # user-code launch related arguments
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=100,
        help="max number of worker group restarts before failing",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=60,
        help="interval (in seconds) to monitor the state of workers",
    )
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="multiprocessing start_method to use when creating workers",
    )
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )

    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args(args)


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[0])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')

    return min_nodes, max_nodes


def get_rdzv_url(
    backend: str, endpoint: str, id: str, min_nodes: int, max_nodes: int, conf: str
):
    # TODO stop relying on init method urls make this a proper factory
    # url exists for historical reasons since we used to rely on torch's c10d interfaces
    # the url format currently ONLY works with etcd
    # TODO note this won't work with zeus because it uses min_size, max_size instead of
    # min_workers, max_workers, change that is zeus.py
    url = f"{backend}://{endpoint}/{id}?min_workers={min_nodes}&max_workers={max_nodes}"

    for kv in conf.split(","):
        if kv:
            conf_key, conf_val = kv.split("=")
            url += f"&{conf_key}={conf_val}"
    return url


def wrapper_fn(omp_num_threads, use_env, cmd):
    # TODO get rid of this wrapper_fn
    # the agent uses multiprocessing.spawn to create nproc_per_node
    # instances of fn, and hence expects fn to be a callable
    # since this launcher deals with user python scripts and executables
    # we wrap the script/executable with this function which Popens
    # the wrapped script/executable. This implies that for each
    # worker we create two processes (wrapper_fn and fn).
    # the process tree looks like the following:
    #
    # [launcher/agent]
    #               |-- [wrapper_fn_0]
    #               |               |-- [fn_0]
    #               |-- [wrapper_fn_1]
    #               |               |-- [fn_1]
    #               |      ...
    #               |      ...
    #               |-- [wrapper_fn_k]
    #               |               |-- [fn_k]
    #

    # set PyTorch distributed related environmental variables
    if omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    if not use_env:
        # LOCAL_RANK is set by the agent
        cmd.append("--local_rank={}".format(os.environ["LOCAL_RANK"]))

    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def main(args=None):
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    args = parse_args(args)

    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts > 0

    rdzv_handler = dist.rendezvous(
        get_rdzv_url(
            args.rdzv_backend,
            args.rdzv_endpoint,
            args.rdzv_id,
            min_nodes,
            max_nodes,
            args.rdzv_conf,
        )
    )

    omp_num_threads = None
    if "OMP_NUM_THREADS" not in os.environ and args.nproc_per_node > 1:
        omp_num_threads = 1
        print(
            f"*****************************************\n"
            f"Setting OMP_NUM_THREADS environment variable for each process to be "
            f"{omp_num_threads} in default, to avoid your system being overloaded, "
            f"please further tune the variable for optimal performance in "
            f"your application as needed. \n"
            f"*****************************************"
        )

    with_python = not args.no_python
    cmd = []
    if with_python:
        cmd = [sys.executable, "-u"]
        if args.module:
            cmd.append("-m")
    else:
        if not args.use_env:
            raise ValueError(
                "When using the '--no_python' flag,"
                " you must also set the '--use_env' flag."
            )
        if args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag"
                " and the '--module' flag at the same time."
            )

    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    spec = WorkerSpec(
        role="default",
        local_world_size=args.nproc_per_node,
        fn=wrapper_fn,
        args=(omp_num_threads, args.use_env, cmd),
        rdzv_handler=rdzv_handler,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
    )
    elastic_agent = LocalElasticAgent(spec, start_method=args.start_method)
    elastic_agent.run(spec.role)


if __name__ == "__main__":
    main()
