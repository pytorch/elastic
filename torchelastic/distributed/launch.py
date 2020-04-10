#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
This module provides similar functionality as ``torch.distributed.launch``,
with the following additional functionalities:

1. Worker failures are handled gracefully by restarting all workers.

2. Worker ``RANK`` and ``WORLD_SIZE`` are assigned automatically.

3. Number of nodes is allowed to change between min and max sizes (elasticity).

**Usage:**

1. Single-node multi-worker (with sidecar etcd server)

::
    >>> python -m torchelastic.distributed.launch
        --with_etcd
        --nnodes=1
        --nproc_per_node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

2. Fault tolerant (fixed sized number of workers, no elasticity).:

::

    >>> python -m torchelastic.distributed.launch
        --nnodes=$NUM_NODES
        --nproc_per_node=$NUM_TRAINERS
        --rdzv_id=$JOB_ID
        --rdzv_backend=etcd
        --rdzv_endpoint=$ETCD_HOST:$ETCD_PORT
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

3. Elastic (``min=1``, ``max=4``):

::

    >>> python -m torchelastic.distributed.launch
        --nnodes=1:4
        --nproc_per_node=$NUM_TRAINERS
        --rdzv_id=$JOB_ID
        --rdzv_backend=etcd
        --rdzv_endpoint=$ETCD_HOST:$ETCD_PORT
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

**Note on etcd**:

For multi-node training you need to specify:

1. ``--rdzv_id``: a unique job id (shared by all nodes participating in the job)
2. ``--rdzv_backend``: an implementation of ``torchelastic.rendevous.RendezvousHandler``
3. ``--rdzv_endpoint``: ``host:port``-style endpoint where the rdzv backend is running.

Currently only ``etcd`` rdzv backend is supported out of the box.
To use ``etcd``, setup an etcd server with the ``v2`` api enabled
(e.g. ``--enable-v2``).

.. warning:: ``EtcdRendezvous`` uses etcd api v2. You MUST enable the v2
             api on the etcd server. Our tests use etcd v3.4.3.

**Definitions:**

1. ``Node`` - Physical instance or container.
    Maps to the unit that the job manager works with.

2. ``Worker`` - A worker in the context of distributed training.

3. ``Worker Group`` - Workers with the same function (e.g. trainers)

4. ``Local Worker Group`` - Subset of the workers in the
    worker group running on the same Node

5. ``RANK`` - rank of the worker within a worker group.

6. ``WORLD_SIZE`` - total number of workers in a worker group.

7. ``LOCAL_RANK`` - rank of the worker within a local worker group

8. ``LOCAL_WORLD_SIZE`` - size of the local worker group

9. ``rdzv_id`` - user defined id that uniquely identifies the worker group
      for a job. This id is used by each node to join as a member of a particular
      worker group.

9. ``rdzv_backend`` - the backend store of rendezvous (e.g. etcd). This is
    typically a strongly consistent key-value store.

10. ``rdzv_endpoint`` - rdzv backend server endpoint in ``host:port`` format.

A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``.
The union of all ``LocalWorkerGroups`` in the nodes in the job comprise the
``WorkerGroup``.

**Environment Variables:**

The following environment variables are made available to you in your
script:

1. ``LOCAL_RANK`` -  local rank

2. ``RANK`` -  global rank

3. ``GROUP_RANK`` - rank of the worker group. A number between 0 - ``max_nnodes``.
        When running a single worker group per node, this is the rank of the node.

4. ``LOCAL_WORLD_SIZE`` - local world size (e.g. number of workers running locally).
       Equal to ``--nproc_per_node`` specified on ``torchelastic.distributed.launch``.

5. ``WORLD_SIZE`` - world size (total number of workers in the job).

6. ``MASTER_ADDR`` - fqdn of the host that is running worker with rank 0.
   Used to initialize torch distributed backend.

7. ``MASTER_PORT`` - port on the ``MASTER_ADDR`` that can be used to
   host the tcp ``c10d`` store.

8. ``TORCHELASTIC_RESTART_COUNT`` - number of worker group restarts so far.

9. ``TORCHELASTIC_MAX_RESTARTS`` - configured max number of restarts.

**Deployment:**

1. Start the rdzv backend server and get the endpoint
   (to be passed as ``--rdzv_endpoint`` to the launcher script)

2. Single-node multi-worker - start the launcher on the host to start
   the agent process which creates and monitors a local worker group.

3.Multi-node multi-worker - Start the launcher with the same arguments
  on all the nodes participating in training.

When using a job/cluster manager the entry point command to the multi-node
job is invoking this launcher.

**Failure Modes:**

1. Worker failure - For a training job with ``n`` workers, if ``k < n`` workers fail
   all workers are stopped and restarted up to ``max_restarts``.

2. Agent failure - An agent failure results in local worker group failure,
   it is up to the job manager to fail the entire job (gang semantics) or attempt
   to replace the node. Both behaviors are supported by the agent.

3. Node failure - Same as agent failure.

**Membership Changes:**

1. Node departure (scale-down) - agent is notified of the departure,
   all existing workers are stopped, a new ``Worker Group`` is formed and all
   workers are started with a new ``RANK`` and ``WORLD_SIZE``.

2. Node arrival (scale-up) - the new node is admitted to the job,
   all existing workers are stopped, a new ``Worker Group`` is formed and all
   workers are started with a new ``RANK`` and ``WORLD_SIZE``.


**Important Notices:**

1. All the items in the important notices section of ``torch.distributed.launch``
   apply to this module as well

2. The environment variables necessary to initialize a torch process group
   are provided to you by this module, no need for you to pass ``RANK`` manually.
   To initialize a process group in your training script, simply run

::

 >>> import torch.distributed as dist
 >>> dist.init_process_group(backend="gloo|nccl")

3. On failures or membership changes ALL surviving workers are killed
   immediately. Make sure to checkpoint your progress. The frequency of
   checkpoints should depend on your job's tolerance for lost work.

4. This module only supports homogeneous ``LOCAL_WORLD_SIZE``. That is,
   it is assumed that all nodes run the same number of local workers (per role).

5. ``RANK`` is NOT stable. Between restarts, the local workers on a node
   can be assgined a different range of ranks than before. NEVER hard code
   any assumptions about the stable-ness of ranks or some correlation between
   ``RANK`` and ``LOCAL_RANK``.

6. When using elasticity (``min_size != max_size``) DO NOT hard code
   assumptions about ``WORLD_SIZE`` as the world size can change due as
   nodes are allowed to leave and join.

7. It is recommended your script have the following structure

::

  def main():
    load_checkpoint(checkpoint_path)
    initialize()
    train()

  def train():
    for batch in iter(dataset):
      train_step(batch)

      if should_checkpoint:
        save_checkpoint(checkpoint_path)
"""
import logging
import os
import signal
import subprocess
import sys
import uuid
from argparse import REMAINDER, ArgumentParser

import torch
import torchelastic.rendezvous.etcd_rendezvous  # noqa: F401
import torchelastic.rendezvous.parameters as parameters
from torchelastic import metrics
from torchelastic.agent.server.api import WorkerSpec
from torchelastic.agent.server.local_elastic_agent import LocalElasticAgent
from torchelastic.rendezvous.etcd_server import EtcdServer


log = logging.getLogger(__name__)


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
        "--nproc_per_node",
        type=str,
        default="auto",
        help="number of workers per node, supported values: [auto, cpu, gpu, int]",
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

    # sidecar etcd related arguments
    parser.add_argument(
        "--with_etcd",
        default=False,
        action="store_true",
        help="starts a local, standalone etcd server on a random free port"
        "using the etcd binary specified in TORCHELASTIC_ETCD_BINARY_PATH"
        " env var or the one found in PATH."
        " Useful when launching single-node, multi-worker job."
        " If specified --rdzv_backend, --rdzv_endpoint, --rdzv_id"
        " are autoassigned, any explicitly set values are ignored",
    )

    # user-code launch related arguments
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=3,
        help="max number of worker group restarts before failing",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=5,
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
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')

    return min_nodes, max_nodes


def wrapper_fn(omp_num_threads, cmd):
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

    process = subprocess.Popen(cmd)

    # since we wrap the script process with this function (which runs as a
    # subprocess of the agent) when the agent terminates this function
    # due to some exception or membership change event we want the script
    # to also get killed. If we do not register this exit handler
    # the script process will get re-parented to the parent of this function
    # (agent process) and we will end up with multiple copies of the script
    # this should all go away with D20613415
    def kill_script_pid(signum, frame):
        process.terminate()

    signal.signal(signal.SIGTERM, kill_script_pid)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


def determine_local_world_size(nproc_per_node: str):
    try:
        logging.info(f"Using nproc_per_node={nproc_per_node}.")
        return int(nproc_per_node)
    except ValueError:
        if nproc_per_node == "cpu":
            num_proc = os.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.")
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == "auto":
            if torch.cuda.is_available():
                num_proc = torch.cuda.device_count()
                device_type = "gpu"
            else:
                num_proc = os.cpu_count()
                device_type = "cpu"
        else:
            raise ValueError(f"Unsupported nproc_per_node value: {nproc_per_node}")

        log.info(
            f"Using nproc_per_node={nproc_per_node},"
            f" seting to {num_proc} since the instance "
            f"has {os.cpu_count()} {device_type}"
        )
        return num_proc


def main(args=None):
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    args = parse_args(args)

    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts > 0

    if args.with_etcd:
        etcd_server = EtcdServer()
        etcd_server.start()
        args.rdzv_backend = "etcd"
        args.rdzv_endpoint = etcd_server.get_endpoint()
        args.rdzv_id = str(uuid.uuid4())
        log.info(
            f"\n**************************************\n"
            f"Rendezvous info:\n"
            f"--rdzv_backend={args.rdzv_backend} "
            f"--rdzv_endpoint={args.rdzv_endpoint} "
            f"--rdzv_id={args.rdzv_id}\n"
            f"**************************************\n"
        )

    rdzv_parameters = parameters.RendezvousParameters(
        args.rdzv_backend,
        args.rdzv_endpoint,
        args.rdzv_id,
        min_nodes,
        max_nodes,
        args.rdzv_conf,
    )

    rdzv_handler = parameters.get_rendezvous(rdzv_parameters)
    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    omp_num_threads = None
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
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
        if args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag"
                " and the '--module' flag at the same time."
            )

    cmd.append(args.training_script)
    cmd.extend(args.training_script_args)

    spec = WorkerSpec(
        role="default",
        local_world_size=nproc_per_node,
        fn=wrapper_fn,
        args=(omp_num_threads, cmd),
        rdzv_handler=rdzv_handler,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
    )
    metrics.initialize_metrics()
    elastic_agent = LocalElasticAgent(spec, start_method=args.start_method)
    elastic_agent.run(spec.role)

    if args.with_etcd:
        etcd_server.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
    )
    log.info(f"Running torchelastic.distributed.launch with args: {sys.argv}")
    main()
