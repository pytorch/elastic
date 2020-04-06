Examples
=========

The examples below run on the `torchelastic/examples <https://hub.docker.com/r/torchelastic/examples>`_
Docker image, built from the `examples/Dockerfile <https://github.com/pytorch/elastic/blob/master/examples/Dockerfile>`_.
The ``$VERSION`` (e.g. ``0.2.0rc0``) variable is used throughout this page,
this should be substitued with the version of torchelastic you are using.

.. note:: the examples below only work on version ``>0.2.0rc0``.

In most cases we demonstrate a single-node, multi-worker example. To use
multiple nodes, run the same commands on multiple nodes with these differences:

1. Specify ``--nnodes=$MIN_NODE:$MAX_NODE`` instead of ``--nnodes=1``.
2. Remove ``--with_etcd`` and specify ``--rdzv_backend``, ``--rdzv_endpoint`` and ``--rdzv_id``.

For more information see `elastic launch <distributed.html>`_).

Alternatively, you can use our kubernetes `elastic job controller <kubernetes.html>`_
to launch a multi-node job.


Prerequisite
--------------

1. `Docker <https://docs.docker.com/install/>`_
2. ``export VERSION=<torchelastic version>``

Classy Vision
--------------
`Classy Vision <https://classyvision.ai/>`_ is an end-to-end framework
for image and video classification built on PyTorch. It works out-of-the-box
with torchelastic's launcher.

Launch two trainers on a single node:

.. code-block:: bash

   docker run torchelastic/examples:$VERSION
              --with_etcd
              --nnodes=1
              --nproc_per_node=2
              /workspace/classy_vision/classy_train.py
              --config_file /workspace/classy_vision/configs/template_config.json

If you have an instance with GPUs, run a worker on each GPU:

.. code-block:: bash

   docker run torchelastic/examples:$VERSION
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

   docker run torchelastic/examples:$VERSION
              --with_etcd
              --nnodes=1
              --nproc_per_node=$NUM_CUDA_DEVICES
              /workspace/examples/imagenet/main.py
              --arch resnet18
              --epochs 20
              --batch-size 32
              /workspace/data/tiny-imagenet-200
