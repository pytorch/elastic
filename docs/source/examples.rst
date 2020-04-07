Examples
=========

The examples below run on the `torchelastic/examples <https://hub.docker.com/r/torchelastic/examples>`_
Docker image, built from the `examples/Dockerfile <https://github.com/pytorch/elastic/blob/master/examples/Dockerfile>`_.

.. note:: The ``$VERSION`` (e.g. ``0.2.0rc0``) variable is used throughout this page,
          this should be substituted with the version of torchelastic you are using.
          The examples below only work on torchelastic ``>0.2.0rc0``.

Prerequisite
--------------

1. (recommended) Instance with GPU(s)
2. `Docker <https://docs.docker.com/install/>`_
3. `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_
4. ``export VERSION=<torchelastic version>``

.. note:: PyTorch data loaders use ``shm``. The default docker ``shm-size`` is not
          large enough and will OOM when using multiple data loader workers.
          You must pass ``--shm-size`` to the ``docker run`` command or set the
          number of data loader workers to ``0`` (run on the same process)
          by passing the appropriate option to the script (use the ``--help`` flag
          to see all script options). In the examples below we set ``--shm-size``.

Classy Vision
--------------
`Classy Vision <https://classyvision.ai/>`_ is an end-to-end framework
for image and video classification built on PyTorch. It works out-of-the-box
with torchelastic's launcher.

Launch two trainers on a single node:

.. code-block:: bash

   docker run --shm-size=2g torchelastic/examples:$VERSION
              --with_etcd
              --nnodes=1
              --nproc_per_node=2
              /workspace/classy_vision/classy_train.py
              --config_file /workspace/classy_vision/configs/template_config.json

If you have an instance with GPUs, run a worker on each GPU:

.. code-block:: bash

   docker run --shm-size=2g --gpus=all torchelastic/examples:$VERSION
              --with_etcd
              --nnodes=1
              --nproc_per_node=$NUM_CUDA_DEVICES
              /workspace/classy_vision/classy_train.py
              --device=gpu
              --config_file /workspace/classy_vision/configs/template_config.json

Imagenet
----------

.. note:: an instance with at least one GPU is required for this example

Launch ``$NUM_CUDA_DEVICES`` number of workers on a single node:

.. code-block:: bash

   docker run --shm-size=2g --gpus=all torchelastic/examples:$VERSION
              --with_etcd
              --nnodes=1
              --nproc_per_node=$NUM_CUDA_DEVICES
              /workspace/examples/imagenet/main.py
              --arch resnet18
              --epochs 20
              --batch-size 32
              /workspace/data/tiny-imagenet-200

Multi-container, multi-worker
-------------------------------

In this example we will launch multiple containers on a single node.
Each container is running multiple workers.
This demonstrates how a multi-node launch would work (each node runs a container).

The high-level differences between a single-container vs multi-container
launches are:

1. Specify ``--nnodes=$MIN_NODE:$MAX_NODE`` instead of ``--nnodes=1``.
2. An etcd server must be setup before starting the worker containers.
3. Remove ``--with_etcd`` and specify ``--rdzv_backend``, ``--rdzv_endpoint`` and ``--rdzv_id``.

For more information see `elastic launch <distributed.html>`_).

<PLACEHOLDER, add multi-container example instructions here>

Multi-node, multi-worker
-------------------------

The multi-node, multi-worker case is similar to running multi-container, multi-worker.
Simply run each container on a separate node, occupying the entire node.
Alternatively, you can use our kubernetes
`elastic job controller <kubernetes.html>`_ to launch a multi-node job.
