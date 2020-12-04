Training Session Manager (TSM)
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

Usage Overview
----------------------

.. automodule:: torchelastic.tsm.driver

User API Documentation
--------------------------

.. currentmodule:: torchelastic.tsm.driver.api

Containers and Resource
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Container
.. autoclass:: Resource

Roles and Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: Role
.. autoclass:: ElasticRole
.. autoclass:: macros
.. autoclass:: Application

Session
~~~~~~~~~~~~~~~
.. autoclass:: Session
   :members:

Developer API Documentation
-----------------------------------
.. currentmodule:: torchelastic.tsm.driver.api

Scheduler
~~~~~~~~~~~~~~~
.. autoclass:: torchelastic.tsm.driver.api.Scheduler
   :members:

.. autoclass:: torchelastic.tsm.driver.local_scheduler.LocalScheduler
   :members:

Session Builder
~~~~~~~~~~~~~~~~
