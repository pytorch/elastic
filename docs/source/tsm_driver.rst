Overview
===============================================

Training Session Manager (TSM) is a set of programmatic APIs that
helps you launch your distributed (PyTorch) applications onto the
supported schedulers. Whereas torchelastic is deployed per container
and manages worker processes and coordinates restart behaviors, TSM
provides a way to launch the distributed job while natively supporting
jobs that are (locally) managed by torchelastic.

.. note:: TSM is currently an experimental module
          and is subject to change for future releases of torchelastic.
          At the moment TSM only ships with a ``LocalScheduler`` allowing
          the user to run the distributed application locally on a dev host.

Usage
----------------------

.. automodule:: torchelastic.tsm.driver
