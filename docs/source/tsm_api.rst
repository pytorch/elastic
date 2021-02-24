.. currentmodule:: torchelastic.tsm.driver.api

Session
======================
.. autoclass:: Session
   :members:

Containers and Resource
======================
.. autoclass:: Container
.. autoclass:: Resource

Roles and Applications
======================
.. autoclass:: Role
.. autoclass:: ElasticRole
.. autoclass:: macros
.. autoclass:: RetryPolicy
.. autoclass:: Application


Status API
======================
.. autoclass:: AppStatus
.. autoclass:: RoleStatus
.. autoclass:: ReplicaStatus


Extending TSM
--------------
TSM is built in a "plug-n-play" manner. While it ships out-of-the-box
with certain schedulers and session implementations, you can implement
your own to fit the needs of your PyTorch application and infrastructure.
This section introduces the interfaces that were meant to be subclassed
and extended.


.. currentmodule:: torchelastic.tsm.driver.api

Scheduler
======================
.. autoclass:: torchelastic.tsm.driver.api.Scheduler
   :members:

.. autoclass:: torchelastic.tsm.driver.local_scheduler.LocalScheduler
   :members:
