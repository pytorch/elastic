#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Below is a simple program that locally launches a multi-role
(trainer, parameter server, reader) distributed application.
Each ``Role`` runs multiple replicas. In reality each replica
runs on its own container on a host. An ``Application`` is made up
to one or more such ``Roles``.

.. code:: python

 import getpass
 import torchelastic.tsm.driver as tsm

 username = getpass.getuser()
 train_project_dir = tsm.Container(image=f"/home/{username}/pytorch_trainer")
 reader_project_dir = tsm.Container(image=f"/home/{username}/pytorch_reader")

 trainer = tsm.ElasticRole(name="trainer", nprocs_per_node=2, nnodes="4:4")
              .runs("train_main.py", "--epochs", "50", MY_ENV_VAR="foobar")
              .on(train_project_dir)
              .replicas(4)

 ps = tsm.Role(name="parameter_server")
         .run("ps_main.py")
         .on(train_project_dir)
         .replicas(10)

 reader = tsm.Role(name="reader")
             .runs("reader/reader_main.py", "--buffer_size", "1024")
             .on(reader_project_dir)
             .replicas(1)

 app = tsm.Application(name="my_train_job").of(trainer, ps, reader)

 session = tsm.session(name="my_session")
 app_id = session.run(app, scheduler="local")
 session.wait(app_id)

In the example above, we have done a few things:

#. Created and ran a distributed training application that runs a total of
   4 + 10 + 1 = 15 containers (just processes since we used a ``local`` scheduler).
#. ``trainer`` run wrapped with TorchElastic.
#. The ``trainer`` and ``ps`` run from the same image (but different containers):
   ``/home/$USER/pytorch_trainer`` and the reader runs from the image:
   ``/home/$USER/pytorch_reader``. The images map to a local directory
   because we are using a local scheduler. For other non-trivial schedulers
   a container could map to a Docker image, tarball, rpm, etc.
#. The main entrypoints are relative to the container image's root dir.
   For example, the trainer runs ``/home/$USER/pytorch_trainer/train_main.py``.
#. Arguments to each role entrypoint are passed as ``*args`` after the entrypoint CMD.
#. Environment variables to each role entrypoint are passed as ``**kwargs``
   after the arguments.
#. The ``session`` object has action APIs on the app (see :class:`Session`).


"""
import getpass
from typing import Optional

from torchelastic.tsm.driver.api import (  # noqa: F401 F403
    AppDryRunInfo,
    AppHandle,
    Application,
    AppState,
    AppStatus,
    Container,
    DescribeAppResponse,
    ElasticRole,
    Resource,
    RetryPolicy,
    Role,
    RunConfig,
    Scheduler,
    SchedulerBackend,
    Session,
    is_terminal,
    macros,
    parse_app_handle,
    runopts,
)
from torchelastic.tsm.driver.schedulers import get_schedulers
from torchelastic.tsm.driver.standalone_session import StandaloneSession


def get_owner() -> str:
    return getpass.getuser()


try:
    from torchelastic.tsm.driver.api_extended import *  # noqa: F401 F403
except ModuleNotFoundError:
    pass


def _gen_session_name():
    return f"tsm_{get_owner()}"


def session(name: Optional[str] = None, backend: str = "standalone", **scheduler_args):
    if backend != "standalone":
        raise ValueError(
            f"Unsupported session backend: {backend}. Supported values: standalone"
        )

    if not name:
        name = _gen_session_name()

    scheduler_args["session_name"] = name
    return StandaloneSession(name=name, schedulers=get_schedulers(**scheduler_args))
