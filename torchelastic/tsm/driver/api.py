#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any, Dict, List, Optional


@dataclass
class Resources:
    """
    Represents resource requirements for a ``Container``.

    Arguments:
            cpu: number of cpu cores (note: not hyper threads)
            gpu: number of gpus
            memMB: MB of ram
            capabilities: additional hardware specs (interpreted by scheduler)
    """

    cpu: int
    gpu: int
    memMB: int
    capabilities: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def copy(original: "Resources", **capabilities):
        """
        Copies a resource and applies new capabilities. If the same capabilities
        are present in the original resource and as parameter, the one from parameter
        will be used.
        """
        res_capabilities = dict(original.capabilities)
        res_capabilities.update(capabilities)
        return Resources(
            cpu=original.cpu,
            gpu=original.gpu,
            memMB=original.memMB,
            capabilities=res_capabilities,
        )


# sentinel value used for cases when resource does not matter (e.g. ignored)
NULL_RESOURCE: Resources = Resources(cpu=-1, gpu=-1, memMB=-1)


@dataclass
class Container:
    """
    Represents the specifications of the container that instances of ``Roles``
    run on. Maps to the container abstraction that the underlying scheduler
    supports. This could be an actual container (e.g. Docker) or a physical
    instance depending on the scheduler.

    An ``image`` is a software bundle that is installed on a ``Container``.
    The container on the scheduler dictates what an image actually is.
    An image could be as simple as a tar-ball or map to a docker image.
    The scheduler typically knows how to "pull" the image given an
    image name (str), which could be a simple name (e.g. docker image) or a url
    (e.g. s3://path/my_image.tar).

    Usage:

    ::

     my_container = Container(image="pytorch/torch:1")
                       .require(Resources(cpu=1, gpu=1, memMB=500))
                       .ports(tcp_store=8080, tensorboard=8081)
    """

    image: str
    resources: Resources = NULL_RESOURCE
    port_map: Dict[str, int] = field(default_factory=dict)

    def require(self, resources: Resources) -> "Container":
        """
        Sets resource requirements on the container.
        """
        self.resources = resources
        return self

    def ports(self, **kwargs: int) -> "Container":
        """
        Adds a port mapping for the container
        """
        self.port_map.update({**kwargs})
        return self


# sentinel value used to represent missing string attributes, such as image or entrypoint
MISSING: str = "<MISSING>"

# sentinel value used as the "zero" element in the container group
NULL_CONTAINER: Container = Container(image=MISSING)


class macros:
    """
    Defines macros that can be used with ``Role.entrypoint`` and ``Role.args``.
    The macros will be substituted at runtime to their actual values.

    Available macros:

    1. ``img_root`` - root directory of the pulled image on the container
    2. ``app_id`` - application id (same as the return value of ``session.run(app)``)
    3. ``replica_id`` - unique id for each instance of a replica of a Role,
                        for instance a role with 3 replicas could have the 0, 1, 2
                        as replica ids. Note that when the container fails and is
                        replaced, the new container will have the same ``replica_id``
                        as the one it is replacing. For instance if node 1 failed and
                        was replaced by the scheduler the replacing node will also
                        have ``replica_id=1``.

    Example:

    ::

     # runs: hello_world.py --app_id ${app_id}
     trainer = Role(name="trainer").runs("hello_world.py", "--app_id", macros.app_id)
     app = Application("train_app").of(trainer)
     app_id = session.run(app)


    """

    img_root = "${img_root}"
    app_id = "${app_id}"
    replica_id = "${replica_id}"

    @staticmethod
    def substitute(args: List[str], img_root: str, app_id: str, replica_id: str):
        args_sub = []
        for arg in args:
            sub = Template(arg).safe_substitute(
                img_root=img_root, app_id=app_id, replica_id=replica_id
            )
            args_sub.append(sub)
        return args_sub


@dataclass
class Role:
    """
    A set of nodes that perform a specific duty within the ``Application``.
    Examples:

    1. Distributed data parallel app - made up of a single role (trainer).

    2. App with parameter server - made up of multiple roles (trainer, ps).

    Usage:

    ::

     trainer = Role(name="trainer")
                 .runs("my_trainer.py", "--arg", "foo", ENV_VAR="FOOBAR")
                 .on(container)
                 .replicas(4)

    Arguments:
            name - name of the role
            entrypoint - command (within the container) to invoke the role
            args - commandline arguments to the entrypoint cmd
            env - environment variable mappings
            container - container to run in
            replicas - number of container replicas to run
    """

    name: str
    entrypoint: str
    args: List[str]
    env: Dict[str, str]
    container: Container
    num_replicas: int

    def __init__(self, name: str):
        self.name = name
        self.entrypoint = MISSING
        self.args = []
        self.env = {}
        self.container = NULL_CONTAINER
        self.num_replicas = 1

    def runs(self, entrypoint: str, *args: str, **kwargs: str) -> "Role":
        self.entrypoint = entrypoint
        self.args += [*args]
        self.env.update({**kwargs})
        return self

    def on(self, container: Container) -> "Role":
        self.container = container
        return self

    def replicas(self, replicas: int) -> "Role":
        self.num_replicas = replicas
        return self


@dataclass
class ElasticRole(Role):
    """
    A ``Role`` for which the user provided ``entrypoint`` is executed with the
    torchelastic agent (in the container). Note that the torchelastic agent
    invokes multiple copies of ``entrypoint``.

    For more information about torchelastic see
    `torchelastic quickstart docs <http://pytorch.org/elastic/0.2.0/quickstart.html>`__.

    .. important:: It is the responsibility of the user to ensure that the
                   container's image includes torchelastic. Since TSM has no
                   control over the build process of the image, it cannot
                   automatically include torchelastic in the container's image.

    The following example launches 2 ``replicas`` (nodes) of an elastic ``my_train_script.py``
    that is allowed to scale between 2 to 4 nodes. Each node runs 8 workers which are allowed
    to fail and restart a maximum of 3 times.

    .. warning:: ``replicas`` MUST BE an integer between (inclusive) ``nnodes``. That is,
                   ``ElasticRole("trainer", nnodes="2:4").replicas(5)`` is invalid and will
                   result in undefined behavior.

    ::

     # effectively runs:
     #    python -m torchelastic.distributed.launch
     #        --nproc_per_node 8
     #        --nnodes 2:4
     #        --max_restarts 3
     #        my_train_script.py --script_arg foo --another_arg bar

     elastic_trainer = ElasticRole("trainer", nproc_per_node=8, nnodes="2:4", max_restarts=3)
                        .runs("my_train_script.py", "--script_arg", "foo", "--another_arg", "bar")
                        .on(container)
                        .replicas(2)

    """

    torchelastic_launch_args: List[str]

    def __init__(self, name: str, **launch_kwargs):
        super().__init__(name)
        self.entrypoint = "python"
        self.args += ["-m", "torchelastic.distributed.launch"]
        self.torchelastic_launch_args = []

        launch_kwargs.setdefault("rdzv_backend", "etcd")
        launch_kwargs.setdefault("rdzv_id", macros.app_id)
        launch_kwargs.setdefault("role", name)

        for (arg, val) in launch_kwargs.items():
            if isinstance(val, bool):
                # treat boolean kwarg as a flag
                if val:
                    self.torchelastic_launch_args += [f"--{arg}"]
            else:
                self.torchelastic_launch_args += [f"--{arg}", str(val)]

    def runs(self, entrypoint: str, *args: str, **kwargs: str) -> "ElasticRole":
        if not os.path.isabs(entrypoint) and not entrypoint.startswith(macros.img_root):
            # make entrypoint relative to {img_root} ONLY if it is not an absolute path
            entrypoint = os.path.join(macros.img_root, entrypoint)

        self.args += self.torchelastic_launch_args
        self.args += [entrypoint, *args]
        self.env.update({**kwargs})
        return self


class RunMode(Enum):
    """
    Run mode of the application. Supported modes are:

    1. HEADLESS - Application is allowed to continue running even when the
                  hosting session has terminated. This is the default mode.

    2. MANAGED - Application's lifetime is bound to the hosting session. If
                 the session goes away, so will the app.
    """

    HEADLESS = 0
    MANAGED = 1


@dataclass
class Application:
    """
    Represents a distributed application made up of multiple ``Roles``.
    Contains the necessary information for the driver to submit this
    app to the scheduler.
    """

    name: str
    roles: List[Role] = field(default_factory=list)
    run_mode: RunMode = RunMode.HEADLESS
    is_attached: bool = False

    def of(self, *roles: Role) -> "Application":
        self.roles += [*roles]
        return self


class AppState(Enum):
    """
    State of the application. An application starts from an initial
    ``UNSUBMITTED`` state and moves through ``SUBMITTED``, ``PENDING``,
    ``RUNNING`` states finally reaching a terminal state:
    ``SUCCEEDED``,``FAILED``, ``CANCELLED``.

    If the scheduler supports preemption, the app moves from a ``RUNNING``
    state to ``PENDING`` upon preemption.

    If the user stops the application, then the application state moves
    to ``STOPPED``, then to ``CANCELLED`` when the job is actually cancelled
    by the scheduler.

    1. UNSUBMITTED - app has not been submitted to the scheduler yet
    2. SUBMITTED - app has been successfully submitted to the scheduler
    3. PENDING - app has been submitted to the scheduler pending allocation
    4. RUNNING - app is running
    5. SUCCEEDED - app has successfully completed
    6. FAILED - app has unsuccessfully completed
    7. CANCELLED - app was cancelled before completing
    """

    UNSUBMITTED = 0
    SUBMITTED = 1
    PENDING = 2
    RUNNING = 3
    SUCCEEDED = 4
    FAILED = 5
    CANCELLED = 6


_TERMINAL_STATES = [AppState.SUCCEEDED, AppState.FAILED, AppState.CANCELLED]


def is_terminal(state: AppState) -> bool:
    return state in _TERMINAL_STATES


@dataclass
class AppStatus:
    """
    The runtime status of the ``Application``.
    """

    state: AppState
    num_restarts: int = 0
    msg: str = ""
    ui_url: Optional[str] = None

    def is_terminal(self) -> bool:
        return is_terminal(self.state)


@dataclass
class DescribeAppResponse:
    """
    Response object returned by ``Scheduler.describe(app)`` API. Contains
    the status and description of the application as known by the scheduler.
    For some schedulers implementations this response object has necessary
    and sufficient information to recreate an ``Application`` object in the
    absence of the hosting ``Session``. For these types of schedulers,
    the user can re-``run()`` the attached application. Otherwise the user
    can only call non-creating methods (e.g. ``wait()``, ``status()``, etc).

    Since this class is a data class and contains many member variables we
    keep the usage simple and provide a no-args constructor and chose to
    access the member vars directly rather than provide accessors.
    """

    app_id: str = "<NOT_SET>"
    state: AppState = AppState.UNSUBMITTED
    num_restarts: int = -1
    msg: str = ""
    ui_url: Optional[str] = None
    # TODO T72035216 add other fields that come back from the scheduler's describe


class Scheduler(abc.ABC):
    """
    An interface abstracting functionalities of a scheduler.
    Implementors need only implement those methods annotated with
    ``@abc.abstractmethod``.
    """

    @abc.abstractmethod
    def submit(self, app: Application, mode: RunMode) -> str:
        """
        Submits the application to be run by the scheduler.

        Returns:
            The application id that uniquely identifies the submitted app.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        """
        Describes the specified application.

        Returns:
            Application description or ``None`` if the app does not exist.
        """
        raise NotImplementedError()

    def exists(self, app_id: str):
        """
        Returns:
            ``True`` if the app exists (was submitted), ``False`` otherwise
        """
        desc = self.describe(app_id)
        return desc is not None

    @abc.abstractmethod
    def _cancel_existing(self, app_id: str) -> None:
        """
        Kills the application. This method will only be called on an
        application that exists.
        """
        raise NotImplementedError()

    def cancel(self, app_id: str) -> None:
        """
        Cancels/kills the application. This method is idempotent within the same
        thread and is safe to call on the same application multiple times.
        However when called from multiple threads/processes on the same app
        the exact semantics of this method depends on the idempotency guarantees
        of the underlying scheduler API.

        .. note:: This method does not block for the application to reach a
                  cancelled state. To ensure that the application reaches a
                  terminal state use the ``wait`` API.
        """
        if self.exists(app_id):
            self._cancel_existing(app_id)
        else:
            # do nothing if the app does not exist
            return


class UnknownAppException(Exception):
    """
    Raised by ``Session`` APIs when either the application does not
    exist or the application is not owned by the session.
    """

    def __init__(self, app_id: str):
        super().__init__(
            f"Unknown app = {app_id}. Did you forget to call session.run() or session.attach()?"
        )


class AppNotReRunnableException(Exception):
    """
    Raised when the application is not re-runnable. That is, one cannot have two run instances
    of the same application. See ``Session.attach()`` for when this is the case.
    """

    def __init__(self, app: Application):
        super().__init__(
            f"App {app.name} is not re-runnable. To re-run attached apps, use a session/scheduler that supports such action."
        )


class Session(abc.ABC):
    """
    Entrypoint and client-facing API for TSM. Has the methods for the user to
    define and act upon ``Applications``. The ``Session`` is stateful and
    represents a logical workspace of the user. It can be backed by a service
    (e.g. TSM server) for persistence or can be standalone with no persistence
    meaning that the ``Session`` lasts only during the duration of the hosting
    process (see the ``attach()`` API for instructions on re-parenting apps
    between sessions).
    """

    def __init__(self, name: str):
        self._name: str = name

    def name(self) -> str:
        """
        Returns:
            The name of this session.
        """
        return self._name

    def run(self, app: Application, mode: RunMode = RunMode.HEADLESS) -> str:
        # input validation
        if not app.roles:
            raise ValueError(
                f"No roles for app: {app.name}. Did you forget to call app.of(roles..)?"
            )

        for role in app.roles:
            if not role.entrypoint:
                raise ValueError(
                    f"No entrypoint for role: {role.name}. Did you forget to call role.runs(entrypoint, args, env)?"
                )
            if role.num_replicas <= 0:
                raise ValueError(
                    f"Non-positive replicas for role: {role.name}. Did you forget to call role.replicas(positive_number)?"
                )
            if role.container == NULL_CONTAINER:
                raise ValueError(
                    f"No container for role: {role.name}. Did you forget to call role.on(container)"
                )
            if role.container.resources == NULL_RESOURCE:
                raise ValueError(
                    f"No resources for container: {role.container.image}. Did you forget to call container.require(resources)"
                )

        return self._run(app, mode)

    @abc.abstractmethod
    def _run(self, app: Application, mode: RunMode = RunMode.HEADLESS) -> str:
        """
        Runs the given application in the specified mode.

        Returns:
            The application id that is used to call other action APIs on the app

        Raises:
            AppNotReRunnableException - if the session/scheduler does not support re-running attached apps
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def status(self, app_id: str) -> Optional[AppStatus]:
        """
        Returns:
            The status of the application, or ``None`` if the app does not exist anymore
            (e.g. was stopped in the past and removed from the scheduler's backend).

        Raises:
            UnknownAppException - if the app was never run or attached on this session
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def wait(self, app_id: str) -> Optional[AppStatus]:
        """
        Block waits (indefinitely) for the application to complete.
        Possible implementation:

        ::

         while(True):
             app_status = status(app)
             if app_status.is_terminal():
                 return
             sleep(10)

        Returns:
            The terminal status of the application, or ``None`` if the app does not exist anymore

        Raises:
            UnknownAppException - if the app was never run or attached on this session
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list(self) -> Dict[str, Application]:
        """
        Returns the applications that were run with this session mapped by the app_id.
        The persistence of the session is implementation dependent.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self, app_id: str) -> None:
        """
        Stops the application, effectively directing the scheduler to cancel
        the job.

        .. note:: This method returns as soon as the cancel request has been
                  submitted to the scheduler. The application will be in a
                  ``RUNNING`` state until the scheduler actually terminates
                  the job. If the scheduler successfully interrupts the job
                  and terminates it the final state will be ``CANCELLED``
                  otherwise it will be ``FAILED``.

        Raises:
            UnknownAppException - if app was never run or attached on this session
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def attach(self, app_id: str) -> Application:
        """
        Attaches the application represented by the provided ``app_id``.
        If the app is already attached to this session, then does nothing
        and returns the existing ``Application`` object.

        Whether all action APIs are supported on the attached application
        is dependent on the specific implementation of the session and scheduler.
        For instance a ``StandaloneSession`` with no persistent backend cannot
        fully recreate the application without the underlying scheduler
        supporting a ``job_definition`` API that stores the job's template.
        Hence in this case, the user will not be able to re-run the attached
        app and can only call action APIs that require only the ``app_id``
        to execute. Please refer to the specific session and scheduler
        implementation documentations to understand the attach behavior,
        assumptions, and consequences.

        Returns:
            The re-created application object

        Raises:
            UnknownAppException - if app was never run or attached on this session
        """
        raise NotImplementedError()
