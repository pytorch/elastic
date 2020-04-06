:github_url: https://github.com/pytorch/elastic

PyTorch Elastic
==================================
PyTorch Elastic enables distributed PyTorch training jobs to be executed
in a fault tolerant and elastic manner.

Use cases:

#. Fault Tolerance: jobs that run on infrastructure where nodes get replaced
   frequently, either due to flaky hardware or by design. Or mission critical
   production grade jobs that need to be run with resilience to failures.

#. Dynamic Capacity Management: jobs that run on leased capacity that can be
   taken away at any time (e.g. AWS spot instances) or shared pools where the
   pool size can change dynamically based on demand.


Quickstart
-----------

.. code-block:: bash

   pip install torchelastic

   # start a single-node etcd server on ONE host
   etcd --enable-v2
        --listen-client-urls http://0.0.0.0:2379,http://127.0.0.1:4001
        --advertise-client-urls PUBLIC_HOSTNAME:2379

To launch a **fault-tolerant** job, run the following on all nodes.

.. code-block:: bash

    python -m torchelastic.distributed.launch
            --nnodes=NUM_NODES
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


To launch an **elastic** job, run the following on at least ``MIN_SIZE`` nodes
and at most ``MAX_SIZE`` nodes.

.. code-block:: bash

    python -m torchelastic.distributed.launch
            --nnodes=MIN_SIZE:MAX_SIZE
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


Train script
-------------

If your train script works with ``torch.distributed.launch`` it will continue
working with ``torchelastic.distributed.launch`` with these differences:

1. No need to manually pass ``RANK``, ``WORLD_SIZE``,
   ``MASTER_ADDR``, and ``MASTER_PORT``.

2. ``rdzv_backend`` and ``rdzv_endpoint`` must be provided. For most users
   this will be set to ``etcd`` (see `rendezvous <rendezvous.html>`_).

3. Make sure you have a ``load_checkpoint(path)`` and
   ``save_checkpoint(path)`` logic in your script. When workers fail
   we restart all the workers with the same program arguments so you will
   lose progress up to the most recent checkpoint
   (see `elastic launch <distributed.html>`_).


Here's an expository example of a training script that checkpoints on each epoch,
hence the max progress lost on failure is one full epoch worth of training.

.. code-block:: python

  def main():
       args = parse_args(sys.argv[1:])
       state = load_checkpoint(args.checkpoint_path)
       initialize(state)

       # torchelastic.distributed.launch ensure that this will work
       # by exporting all the env vars needed to initialize the process group
       torch.distributed.init_process_group(backend=args.backend)

       for i in range(state.epoch, state.total_num_epochs)
            for batch in iter(state.dataset)
                train(batch, state.model)

            state.epoch += 1
            save_checkpoint(state)


Documentation
---------------
.. toctree::
   :maxdepth: 1
   :caption: API

   distributed
   agent
   timer
   rendezvous
   metrics

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples
   kubernetes
   runtime

