:github_url: https://github.com/pytorch/elastic

TorchElastic
==================

TorchElastic enables distributed PyTorch training jobs to be executed
in a fault tolerant and elastic manner.

Use cases:

#. Fault Tolerance: jobs that run on infrastructure where nodes get replaced
   frequently, either due to flaky hardware or by design. Or mission critical
   production grade jobs that need to be run with resilience to failures.

#. Dynamic Capacity Management: jobs that run on leased capacity that can be
   taken away at any time (e.g. AWS spot instances) or shared pools where the
   pool size can change dynamically based on demand.

Documentation
---------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   quickstart
   train_script
   examples

.. toctree::
   :maxdepth: 1
   :caption: API

   distributed
   agent
   timer
   rendezvous
   metrics
   events

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   customization

.. toctree::
   :maxdepth: 1
   :caption: Plugins

   kubernetes
   runtime
