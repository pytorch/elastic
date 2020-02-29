Rendezvous
==========

In the context of torchelastic we use the term “rendezvous” to refer to
a particular functionality that combines a **distributed
synchronization** primitive with **peer discovery**.

It is used by torchelastic to gather participants of a training job
(i.e. workers) such that they all agree on the same list of participants
and everyone’s roles, as well as make a consistent collective decision
on when training can begin/resume.

Torchelastic Rendezvous provides the following critical functionalities:

Barrier
-------

Workers performing rendezvous will all block until the rendezvous is
considered complete - this happens when at least ``min`` total number of
workers have joined the rendezvous barrier (for the same job). This also
implies the barrier is not necessarily of fixed size.

There’s an additional small waiting time after reaching ``min`` number
of workers - this is used to ensure the rendezvous is not completed “too
quickly” (which could potentially exclude additional workers attempting
to join at approximately the same time).

If ``max`` number of workers is gathered at the barrier, the rendezvous
is completed immediately.

There’s also an overall timeout which causes the rendezvous to fail if
``min`` number of workers is never reached – this is meant to be a
simple fail-safe to help release partially allocated job resources, in
case there’s a problem with the resource manger, and is meant to be
interpreted as non-retryable.

Exclusivity
-----------

A simple distributed barrier would not be sufficient, as we also need to
ensure that only one group of workers exists at any given time (for a
given job). In other words, new workers (i.e. joining late) should not
be able to form a parallel independent group of workers for the same
job.

Torchelastic rendezvous ensures that if a group of workers has already
completed a rendezvous (and hence might already be training), then
additional “late” workers attempting to rendezvous will only announce
themselves as waiting, and will have to wait until the (previously
completed) existing rendezvous is destroyed first.

Consistency
-----------

When a rendezvous is completed, all its members will agree on the job
membership and everyone’s role in it. This role is represented using an
integer, called rank, that is between between 0 and world size.

Note that ranks are *not stable*, in the sense that the same worker
process can be assigned a different rank in the next (re-)rendezvous.

Fault-tolerance
---------------

Torchelastic rendezvous is designed to tolerate worker failures during
the rendezvous process. Should a process crash (or lose network
connectivity, etc), between joining the rendezvous and it being
completed, then a re-rendezvous with remaining healthy workers will
happen automatically.

A worker can also fail *after* it has completed (or *has been
observered* by other workers to have completed) the rendezvous - this
scenario will be handled by the torchelastic ``train_loop`` instead
(where it will also trigger a re-rendezvous).

Shared key-value store
----------------------

When the rendezvous is completed, a shared key-value store is created
and returned. This store implements a ``torch.distributed.Store`` API
(see `distributed communication
docs <https://pytorch.org/docs/stable/distributed.html>`__).

This store is only shared by the members of the completed rendezvous. It
is intended to be used by torchelastic to exchange information necessary
to initialize job control and data-planes.

Waiting workers and rendezvous closing
--------------------------------------

Torchelastic rendezvous handler object provides additional
functionalities, which are technically not part of the rendezvous
process: \* Querying how many workers arrived late at the barrier, who
can participate in *next* rendezvous. \* Setting the rendezvous *closed*
to signal all workers not to participate in next rendezvous.

Interface:
----------

Torchelastic rendezvous has the following interface: **WARNING:**
torchelastic is currently considered experimental, so the APIs may
change!

.. code:: python

   class RendezvousHandler(abc.ABC):
       @abc.abstractmethod
       def next_rendezvous(self) -> Tuple["torch.distributed.Store", int, int]:
           pass

       @abc.abstractmethod
       def is_closed(self) -> bool:
           pass

       @abc.abstractmethod
       def set_closed(self):
           pass

       @abc.abstractmethod
       def num_nodes_waiting(self) -> int:
           pass

   class RendezvousClosedException(Exception):
       pass

   class RendezvousTimeoutException(Exception):
       pass

   class RendezvousNonRetryableError(Exception):
       pass

The ``next_rendezvous`` is the main entry-point into the rendezvous
barrier. It blocks until the rendezvous is complete (and the current
process is included in the formed worker group), or a timeout occurs, or
rendezvous was marked closed.

Retuned value is a triplet ``(store, rank, world_size)``. If a timeout
occurs, ``RendezvousTimeoutException`` is raised. If the rendezvous for
current job is closed, ``RendezvousClosedException`` is raised.

``is_closed`` checks whether rendezvous for current job has been closed,
which means all future attempts to re-rendezvous (within same job) will
fail.

``set_closed`` is used to mark the rendezvous (for current job) as
closed.

Note that ``is_closed``/``set_closed`` have semantics of eventual
propagation, and should not be used for synchronization. The intention
here is that if at least one worker decides the job is finished, it will
close the rendezvous, and other workers will “soon” observe this and
stop training/rendezvous-ing as well.

``num_nodes_waiting`` returns number of workers who *arrived late* at
the rendezvous barrier, hence weren’t included in the current worker
group. Torchelastic ``train_loop`` will periodically check
``num_nodes_waiting``, and may decide to pause training in order to
re-rendezvous and include these additional workers.

**NOTE:** Torchelastic users normally **do not** need to implement their
own ``RendezvousHandler``. An implementation based on
`etcd <https://etcd.io/>`__ is already provided, and is recommended for
most users, provided they can deploy it in their environment.

Etcd Rendezvous
---------------

The ``etcd_rendezvous`` implementation in torchelastic uses
`etcd <https://etcd.io/>`__ as the backend store. You can see the full
implementation in `etcd_rendezvous.py <etcd_rendezvous.py>`__. Below is
a state diagram of how it works, |etcd rendezvous state diagram|

Torchelastic uses a URL to configure the type of rendezvous to use and
to pass implementation specific configurations to the rendezvous module.
The basic etcd rendezvous configuration URL looks like the following

::

   etcd://<etcd_address>:<port>/<job_id>?min_workers=<min_workers>&max_workers=<max_workers>

   -- example --

   etcd://localhost:2379/1234?min_workers=1&max_workers=3

The URL above is passed to the constructor of the ``Coordinator`` and it
is interpreted as the following:

1. Use the rendezvous handler that is registered with the ``etcd``
   scheme
2. The ``etcd`` endpoint to use is ``localhost:2379``
3. ``job_id == 1234`` is used as the prefix in etcd (this allows one to
   share a common etcd server for multiple jobs so long as the
   ``job_ids`` are guaranteed to be unique). Note that the job id can be
   any string (e.g. does not need to be a number) as long as it is
   unique.
4. ``min_workers=1`` and ``max_workers=3`` specifies a range for
   membership size - torchelastic starts running the job as long as the
   cluster size is greater than or equal to ``min_workers`` and admits
   up to ``max_workers`` into the cluster.

Below are a full list of the parameters that can be passed to etcd
rendezvous

+--------------------------------------------+--------------------------+
| Parameter                                  | Description              |
+============================================+==========================+
| min_workers                                | minimum number of        |
|                                            | workers for the          |
|                                            | rendezvous to be valid   |
+--------------------------------------------+--------------------------+
| max_workers                                | maximum number of        |
|                                            | workers to admit         |
+--------------------------------------------+--------------------------+
| timeout                                    | total timeout within     |
|                                            | which next_rendezvous is |
|                                            | expected to succeed      |
|                                            | (default 600s)           |
+--------------------------------------------+--------------------------+
| last_call_timeout                          | additional wait amount   |
|                                            | (“last call”) after min  |
|                                            | number of workers has    |
|                                            | been reached (defaults   |
|                                            | to 30s)                  |
+--------------------------------------------+--------------------------+
| etcd_prefix                                | path prefix (from etcd   |
|                                            | root), inside which all  |
|                                            | etcd nodes will be       |
|                                            | created (defaults to     |
|                                            | ``/torchelastic/p2p``)   |
+--------------------------------------------+--------------------------+

Custom Rendezvous
-----------------

You must do the following to implement and use a custom rendezvous
implementation,

1. Implement the `RendezvousHandler <api.py>`__ interface.
2. Register the custom handler with
   ``torch.distributed.register_rendezvous_handler()``
3. Ensure that the registration happens before any calls to load the
   rendezvous object.

For an example, refer to
```etcd_rendezvous.py`` <etcd_rendezvous.py>`__.

.. |etcd rendezvous state diagram| image:: _static/img/etcd_rdzv_diagram.png
