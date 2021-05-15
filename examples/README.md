Examples
=============

The examples below run on the [torchelastic/examples](https://hub.docker.com/r/torchelastic/examples)
Docker image, built from the [examples/Dockerfile](https://github.com/pytorch/elastic/blob/master/examples/Dockerfile).

.. note:: The ``$VERSION`` (e.g. ``0.2.0``) variable is used throughout this page,
          this should be substituted with the version of torchelastic you are using.
          The examples below only work on torchelastic ``>=0.2.0``.

Prerequisite
--------------

1. (recommended) Instance with GPU(s)
2. [Docker](https://docs.docker.com/install/)
3. [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
4. ``export VERSION=<torchelastic version>``

> **NOTE:** PyTorch data loaders use ``shm``. The default docker ``shm-size``
> is not large enough and will OOM when using multiple data loader workers.
> you must pass ``--shm-size`` to the ``docker run`` command or set the
> number of data loader workers to ``0`` (run on the same process)
> by passing the appropriate option to the script (use the ``--help`` flag
> to see all script options). In the examples below we set ``--shm-size``.

Classy Vision
--------------
[Classy Vision](https://classyvision.ai/) is an end-to-end framework
for image and video classification built on PyTorch. It works out-of-the-box
with torchelastic's launcher.

Launch two trainers on a single node:

```
>>> docker run --shm-size=2g torchelastic/examples:$VERSION
               --standalone
               --nnodes=1
               --nproc_per_node=2
               /workspace/classy_vision/classy_train.py
               --config_file /workspace/classy_vision/configs/template_config.json
```

If you have an instance with GPUs, run a worker on each GPU:

```
>>> docker run --shm-size=2g
               --gpus=all
               torchelastic/examples:$VERSION
               --standalone
               --nnodes=1
               --nproc_per_node=$NUM_CUDA_DEVICES
               /workspace/classy_vision/classy_train.py
               --device=gpu
               --config_file /workspace/classy_vision/configs/template_config.json
```

Imagenet
----------

> **NOTE:** an instance with at least one GPU is required for this example

Launch ``$NUM_CUDA_DEVICES`` number of workers on a single node:

```
>>> docker run --shm-size=2g --gpus=all torchelastic/examples:$VERSION
               --standalone
               --nnodes=1
               --nproc_per_node=$NUM_CUDA_DEVICES
               /workspace/examples/imagenet/main.py
               --arch resnet18
               --epochs 20
               --batch-size 32
               /workspace/data/tiny-imagenet-200
```

Multi-container
----------------
We now show how to use the PyTorch Elastic Trainer launcher
to start a distributed application spanning more than one container. The
application is intentionally kept "bare bones" since the
objective is to show how to create a ``torch.distributed.ProcessGroup``
instance. Once a ``ProcessGroup`` is created, you can use any
functionality needed from the ``torch.distributed`` package.

The ``docker-compose.yml`` file is based on the example provided with
the [Bitnami ETCD container image](https://hub.docker.com/r/bitnami/etcd/).



### Obtaining the example repo

Clone the PyTorch Elastic Trainer Git repo using

```
git clone https://github.com/pytorch/elastic.git
```

make an environment variable that points to the elastic repo, e.g.

```
export TORCHELASTIC_HOME=~/elastic
```

### Building the samples Docker container

While you can run the rest of this example using a pre-built Docker
image, you can also build one for yourself. This is especially useful if
you would like to customize the image. To build the image, run:

```
cd $TORCHELASTIC_HOME && docker build -t hello_elastic:dev .
```

### Running an existing sample

This example uses ``docker-compose`` to run two containers: one for the
ETCD service and one for the sample application itself. Docker compose
takes care of all aspects of establishing the network interfaces so the
application container can communicate with the ETCD container.

To start the example, run

```
cd $TORCHELASTIC_HOME/examples/multi_container && docker-compose up
```

You should see two sets of outputs, one from ETCD starting up and one
from the application itself. The output from the application looks
something like this:

```
   example_1      | INFO 2020-04-03 17:36:31,582 Etcd machines: ['http://etcd-server:2379']
   example_1      | *****************************************
   example_1      | Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
   example_1      | *****************************************
   example_1      | INFO 2020-04-03 17:36:31,922 Attempting to join next rendezvous
   example_1      | INFO 2020-04-03 17:36:31,929 New rendezvous state created: {'status': 'joinable', 'version': '1', 'participants': []}
   example_1      | INFO 2020-04-03 17:36:32,032 Joined rendezvous version 1
```

The high-level differences between a single-container vs multi-container
launches are:

1. Specify ``--nnodes=$MIN_NODE:$MAX_NODE`` instead of ``--nnodes=1``.
2. An etcd server must be setup before starting the worker containers.
3. Remove ``--standalone`` and specify ``--rdzv_backend``, ``--rdzv_endpoint`` and ``--rdzv_id``.

For more information see [torch.distributed.run](https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py).



Multi-node
-----------

The multi-node, multi-worker case is similar to running multi-container, multi-worker.
Simply run each container on a separate node, occupying the entire node.
Alternatively, you can use our kubernetes
[elastic job controller](https://github.com/pytorch/elastic/tree/master/kubernetes) to launch a multi-node job.

> **WARNING**: We recommend you setup a highly available etcd server when
> deploying multi-node jobs in production as this is the single
> point of failure for your jobs. Depending on your usecase
> you can either sidecar an etcd server with each job or setup
> a shared etcd server. If etcd does not meet your requirements
> you can implement your own rendezvous handler and use our
> APIs to create a custom launcher.
