#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from enum import Enum
from typing import Any, Dict, List, Optional


class BaseObject:
    """
    Base object class that all object APIs inherit from. Standardizes
    things like serialization, versioning, to_string methods.

    # TODO add serde + versioning methods for robust session.attach(app_id)
    """

    def __repr__(self):
        str = []
        for (field, value) in self.__dict__.items():
            str.append(f"{field}:{value.__repr__()}")
        return f"{{{', '.join(str)}}}"


class Resources(BaseObject):
    """
    Represents resource requirements for a ``Container``.
    """

    __slots__ = ("cpu", "gpu", "memMB", "capabilities")

    def __init__(
        self,
        cpu: int,
        gpu: int,
        memMB: int,
        capabilities: Optional[Dict[str, Any]] = None,
    ):
        """
        Arguments:
            cpu: number of cpus
            gpu: number of gpus
            memMB: MB of ram
            metadata: additional hardware specs (interpreted by scheduler)
        """
        self.cpu: int = cpu
        self.gpu: int = gpu
        self.memMB: int = memMB
        self.capabilities: Optional[Dict[str, Any]] = capabilities


# used for cases when resource does not matter (e.g. ignored)
NULL_RESOURCE: Resources = Resources(cpu=-1, gpu=-1, memMB=-1)


class Container(BaseObject):
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

    def __init__(self, image: str):
        self.image: str = image
        self.resources: Resources = NULL_RESOURCE
        self.port_map: Dict[str, int] = {}

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


# used as the "zero" element in the container group
NULL_CONTAINER: Container = Container(image="<NULL_IMAGE>")


class Role(BaseObject):
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
    """

    def __init__(self, name: str):
        """
        Arguments:
            name - name of the role
            entrypoint - command (within the container) to invoke the role
            args - commandline arguments to the entrypoint cmd
            env - environment variable mappings
            container - container to run in
            replicas - number of container replicas to run
        """
        self.name: str = name
        self.entrypoint: str = ""
        self.args: List[str] = []
        self.env: Dict[str, str] = {}
        self.container: Container = NULL_CONTAINER
        self.num_replicas: int = 1

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


class Application(BaseObject):
    """
    Represents a distributed application made up of multiple ``Roles``.
    Contains the necessary information for the driver to submit this
    app to the scheduler.
    """

    def __init__(self, name: str, run_mode: RunMode = RunMode.HEADLESS):
        self.name = name
        self.roles: List[Role] = []
        self.run_mode = run_mode
        self.is_attached: bool = False

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


class AppStatus:
    """
    The runtime status of the ``Application``.
    """

    def __init__(self, state: AppState, num_restarts: int = 0, msg: str = ""):
        self.state: AppState = state
        self.num_restarts: int = num_restarts
        self.msg: str = msg

    def is_terminal(self) -> bool:
        return is_terminal(self.state)


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

    def __init__(self):
        self.app_id: str = "<NOT_SET>"
        self.state: AppState = AppState.UNSUBMITTED
        self.num_restarts: int = -1
        # TODO add other fields that come back from the scheduler's describe
        # API. Typically this includes some type of JobDefinition which
        # contains all the data to recreate the Application object. Store them
        # here as member vars so that we can implement the session.attach() api:
        #
        # app_id = session.run(app_orig)
        # /* session is lost, create a new session */
        # new_session = Session(...)
        # /* upon attaching the app, app is recreated */
        # app_new = session.attach(app_id)
        # assert app_new == app_orig


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

        # object APIs
        self.app = Application
        self.role = Role
        self.container = Container

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
