#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from string import Template
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

_APP_STATUS_FORMAT_TEMPLATE = """
State: ${state} ; Num Restarts: ${num_restarts}
Msg: ${msg}
Replicas: ${replicas}
"""


_ROLE_REPLICA_FORMAT_TEMPLATE = """
- Role: [${role}]:
${replicas}
"""

_REPLICA_FORMAT_TEMPLATE = """
- [${role}:${replica_id}]
  Timestamp: ${timestamp}; Exit Code: ${exit_code}
  State: ${state}
  Error Message: ${error_msg}
"""


SchedulerBackend = str


@dataclass
class Resource:
    """
    Represents resource requirements for a ``Container``.

    Args:
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
    def copy(original: "Resource", **capabilities):
        """
        Copies a resource and applies new capabilities. If the same capabilities
        are present in the original resource and as parameter, the one from parameter
        will be used.
        """
        res_capabilities = dict(original.capabilities)
        res_capabilities.update(capabilities)
        return Resource(
            cpu=original.cpu,
            gpu=original.gpu,
            memMB=original.memMB,
            capabilities=res_capabilities,
        )


# sentinel value used for cases when resource does not matter (e.g. ignored)
NULL_RESOURCE: Resource = Resource(cpu=-1, gpu=-1, memMB=-1)

# used as "*" scheduler backend
ALL: SchedulerBackend = "all"


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

    A ``Resource`` can be bound to a specific scheduler backend or ``SchedulerBackend.ALL`` (default)
    to specify that the same ``Resource`` is to be used for all schedulers.

    Usage:

    ::

     # define resource for all schedulers
     my_container = Container(image="pytorch/torch:1")
                       .require(Resource(cpu=1, gpu=1, memMB=500))
                       .ports(tcp_store=8080, tensorboard=8081)

     # define resource for a specific scheduler
     my_container = Container(image="pytorch/torch:1")
                       .require(Resource(cpu=1, gpu=1, memMB=500), "custom_scheduler")
                       .ports(tcp_store=8080, tensorboard=8081)

    """

    image: str
    resources: Resource = NULL_RESOURCE
    port_map: Dict[str, int] = field(default_factory=dict)

    def require(self, resources: Resource) -> "Container":
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
    2. ``app_id`` - application id as assigned by the scheduler
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
     app_handle = session.run(app, scheduler="local", cfg=RunConfig())

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


class RetryPolicy(str, Enum):
    """
    Defines the retry policy for the ``Roles`` in the ``Application``.
    The policy defines the behavior when the role replica encounters a failure:

    1. unsuccessful (non zero) exit code
    2. hardware/host crashes
    3. preemption
    4. eviction

    .. note:: Not all retry policies are supported by all schedulers.
              However all schedulers must support ``RetryPolicy.APPLICATION``.
              Please refer to the scheduler's documentation for more information
              on the retry policies they support and behavior caveats (if any).

    1. REPLICA: Replaces the replica instance. Surviving replicas are untouched.
                Use with ``ElasticRole`` to have torchelastic coordinate restarts
                and membership changes. Otherwise, it is up to the application to
                deal with failed replica departures and replacement replica admittance.
    2. APPLICATION: Restarts the entire application.

    """

    REPLICA = "REPLICA"
    APPLICATION = "APPLICATION"


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

    Args:
            name: name of the role
            entrypoint: command (within the container) to invoke the role
            args: commandline arguments to the entrypoint cmd
            env: environment variable mappings
            container: container to run in
            replicas: number of container replicas to run
            max_retries: max number of retries before giving up
            retry_policy: retry behavior upon replica failures
            deployment_preference: hint to the scheduler on how to best
                                   deploy and manage replicas of this role

    """

    name: str
    entrypoint: str = MISSING
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    container: Container = NULL_CONTAINER
    num_replicas: int = 1
    max_retries: int = 0
    retry_policy: RetryPolicy = RetryPolicy.APPLICATION

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

    def with_retry_policy(self, retry_policy: RetryPolicy, max_retries: int) -> "Role":
        self.retry_policy = retry_policy
        self.max_retries = max_retries
        return self


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

     elastic_trainer = ElasticRole("trainer", nproc_per_node=8, nnodes="2:4", max_restarts=3)
                        .runs("my_train_script.py", "--script_arg", "foo", "--another_arg", "bar")
                        .on(container)
                        .replicas(2)
     # effectively runs:
     #    python -m torchelastic.distributed.launch
     #        --nproc_per_node 8
     #        --nnodes 2:4
     #        --max_restarts 3
     #        my_train_script.py --script_arg foo --another_arg bar

    """

    def __init__(self, name: str, **launch_kwargs):
        super().__init__(name=name)
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


@dataclass
class Application:
    """
    Represents a distributed application made up of multiple ``Roles``.
    Contains the necessary information for the driver to submit this
    app to the scheduler.
    """

    name: str
    roles: List[Role] = field(default_factory=list)

    def of(self, *roles: Role) -> "Application":
        self.roles += [*roles]
        return self


class AppState(int, Enum):
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

    UNSUBMITTED = 2 ** 0
    SUBMITTED = 2 ** 1
    PENDING = 2 ** 2
    RUNNING = 2 ** 3
    SUCCEEDED = 2 ** 4
    FAILED = 2 ** 5
    CANCELLED = 2 ** 6

    def __str__(self) -> str:
        return self.name


_TERMINAL_STATES = [AppState.SUCCEEDED, AppState.FAILED, AppState.CANCELLED]


def is_terminal(state: AppState) -> bool:
    return state in _TERMINAL_STATES


NONE: str = "<NONE>"


@dataclass
class RoleReplicaStatus:
    """
    The status of the replica during the job execution.

    Args:
        replica_id: The node rank, note: this is not a worker rank.
        state: The current state of the node.
        exit_code: `None`` if still running
        role: The role name
        end_time: Timestamp value if the node finished execution, None otherwise
        error_msg: Error message if any, None if job succeeded.
    """

    replica_id: int
    state: AppState
    role: str
    exit_code: Optional[int] = None
    end_time: Optional[int] = None
    error_msg: Optional[str] = None

    def get_formatted_str(self) -> str:
        """
        Return human readable status representation.
        """
        return Template(_REPLICA_FORMAT_TEMPLATE).substitute(
            timestamp=self.end_time,
            replica_id=self.replica_id,
            exit_code=self.exit_code,
            state=self.state,
            role=self.role,
            error_msg=self.error_msg,
        )


@dataclass
class AppStatus:
    """
    The runtime status of the ``Application``. The scheduler can
    return an arbitrary text message (msg field).
    If any error occurs, scheduler can populate ``structured_error_msg``
    with json response.

    ``replicas`` represent the statuses of the replicas in the job. If the job
    runs with multiple retries, the parameter will contain the statuses of the
    most recent retry. Note: if the previous retries failed, but the most recent
    retry succeeded or in progress, ``replicas`` will not contain ocurred errors.
    """

    state: AppState
    num_restarts: int = 0
    msg: str = ""
    structured_error_msg: str = NONE
    ui_url: Optional[str] = None
    replicas: Dict[str, List[RoleReplicaStatus]] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def __repr__(self):
        app_status_dict = asdict(self)
        structured_error_msg = app_status_dict.pop("structured_error_msg")
        if structured_error_msg != NONE:
            structured_error_msg_parsed = json.loads(structured_error_msg)
        else:
            structured_error_msg_parsed = NONE
        app_status_dict["structured_error_msg"] = structured_error_msg_parsed
        return json.dumps(app_status_dict, indent=2)

    def _get_role_replicas(
        self, state_mask_filter: int = 0xFF
    ) -> Dict[str, List[RoleReplicaStatus]]:
        filterred_replicas = {}
        for role, role_replicas in self.replicas.items():
            filterred_replicas[role] = [
                replica
                for replica in role_replicas
                if replica.state.value | state_mask_filter == state_mask_filter
            ]
        return filterred_replicas

    def get_formatted_str(self, state_mask_filter: int = 0xFF) -> str:
        """
        Return a human readable representation of the AppStatus.
        """
        role_replicas = ""
        filterred_replicas = self._get_role_replicas(state_mask_filter)
        for role, filterred_role_replicas in filterred_replicas.items():
            if len(filterred_role_replicas) == 0:
                continue
            replicas_str = "".join(
                replica.get_formatted_str() for replica in filterred_role_replicas
            )
            role_replicas += Template(_ROLE_REPLICA_FORMAT_TEMPLATE).substitute(
                role=role, replicas=replicas_str
            )
        return Template(_APP_STATUS_FORMAT_TEMPLATE).substitute(
            state=self.state,
            num_restarts=self.num_restarts,
            msg=self.msg,
            replicas=role_replicas,
        )


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

    If scheduler returns arbitrary message, the ``msg`` field should be populated.
    If scheduler returns a structured json, the ``structured_error_msg`` field should be populated.
    """

    app_id: str = "<NOT_SET>"
    state: AppState = AppState.UNSUBMITTED
    num_restarts: int = -1
    msg: str = NONE
    structured_error_msg: str = NONE
    ui_url: Optional[str] = None

    roles: List[Role] = field(default_factory=list)
    replicas: Dict[str, List[RoleReplicaStatus]] = field(default_factory=dict)


# valid ``RunConfig`` values; only support primitives (str, int, float, bool)
ConfigValue = Union[str, int, float, bool, None]


@dataclass(frozen=True)
class RunConfig:
    """
    Additional run configs for the app. These are typically
    scheduler runtime configs/arguments that do not bind
    to ``Application`` nor the ``Scheduler``. For example
    a particular cluster (within the scheduler) the application
    should be submitted to. Since the same app can be launched
    into multiple types of clusters (devo, prod) the
    cluster id config does not bind to the app. Neither
    does this bind to the scheduler since the cluster can
    be partitioned by size of the instances (S, M, L) or by
    a preemption setting (e.g. on-demand vs spot).

    Since ``Session`` allows the application to be submitted
    to multiple schedulers, users who want to submit the same
    app into multiple schedulers from the same session can
    union all the ``RunConfig``s into a single object. The
    scheduler implementation will selectively read the configs
    it needs.

    This class is intended to be trivially serialized and
    passed around or saved hence only allow primitives
    as config values. Should the scheduler need more than
    simple primitives (e.g. list of str) it is up to the
    scheduler to document a way to encode thie value as a
    str and parse it (e.g. representing list of str as
    comma delimited str).

    Usage:

    ::

     # write
     config = RunConfig()
     config.set("run_as_user", prod")
     config.set("priority", 10)

     # read
     config.get("run_as_user") # "prod"
     config.get("priority") # 10
     config.get("never_set") # None
    """

    cfgs: Dict[str, ConfigValue] = field(default_factory=dict)

    def set(self, cfg_key: str, cfg_val: ConfigValue):
        self.cfgs[cfg_key] = cfg_val

    def get(self, key: str) -> ConfigValue:
        return self.cfgs.get(key, None)

    def __repr__(self):
        return self.cfgs.__repr__()


T = TypeVar("T")


class AppDryRunInfo(Generic[T]):
    """
    Returned by ``Scheduler.submit_dryrun``. Represents the
    request that would have been made to the scheduler.
    The ``fmt_str()`` method of this object should return a
    pretty formatted string representation of the underlying
    request object such that ``print(info)`` yields a human
    readable representation of the underlying request.
    """

    def __init__(self, request: T, fmt: Callable[[T], str]):
        self.request = request
        self._fmt = fmt

        # fields below are only meant to be used by
        # Scheduler or Session implementations
        # and are back references to the parameters
        # to dryrun() that returned this AppDryRunInfo object
        # thus they are set in Session.dryrun() and Scheduler.submit_dryrun()
        # manually rather than through constructor arguments
        # DO NOT create getters or make these public
        # unless there is a good reason to
        self._app = None
        self._cfg = None
        self._scheduler = None

    def __repr__(self):
        return self._fmt(self.request)


class runopts:
    """
    Holds the accepted scheduler run configuration
    keys, default value (if any), and help message string.
    These options are provided by the ``Scheduler`` and validated
    in ``Session.run`` against user provided ``RunConfig``.
    Allows ``None`` default values. Required opts must NOT have a
    non-None default.

    .. important:: This class has no accessors because it is intended to
                   be constructed and returned by ``Scheduler.run_config_options``
                   and printed out as a "help" tool or as part of an exception msg.
    Usage:

    ::
     opts = runopts()

     opts.add("run_as_user", type_=str, help="user to run the job as")
     opts.add("cluster_id", type_=int, help="cluster to submit the job", required=True)
     opts.add("priority", type_=float, default=0.5, help="job priority")
     opts.add("preemptible", type_=bool, default=False, help="is the job preemptible")

     # invalid
     opts.add("illegal", default=10, required=True)
     opts.add("bad_type", type=str, default=10)

     opts.check(RunConfig)
     print(opts)
    """

    def __init__(self):
        self._opts: Dict[str, Tuple[ConfigValue, Type[ConfigValue], bool, str]] = {}

    def add(
        self,
        cfg_key: str,
        type_: Type[ConfigValue],
        help: str,
        default: ConfigValue = None,
        required=False,
    ):
        """
        Adds the ``config`` option with the given help string and ``default``
        value (if any). If the ``default`` is not specified then this option
        is a required option.
        """
        if required and default is not None:
            raise ValueError(
                f"Required option: {cfg_key} must not specify default value. Given: {default}"
            )
        if default is not None:
            if not isinstance(default, type_):
                raise TypeError(
                    f"Option: {cfg_key}, must be of type: {type_}."
                    f" Given: {default} ({type(default).__name__})"
                )

        self._opts[cfg_key] = (default, type_, required, help)

    def resolve(self, config: RunConfig):
        """
        Checks the given config against this ``runopts`` and sets default configs
        if not set.

        .. warning:: This method mutates the provided config!

        """

        # make a copy; don't need to be deep b/c the values are primitives
        resolved_cfg = RunConfig(config.cfgs.copy())

        for cfg_key, (default, type_, required, _help) in self._opts.items():
            val = resolved_cfg.get(cfg_key)

            # check required opt
            if required and val is None:
                raise InvalidRunConfigException(
                    f"Required run option: {cfg_key}, must be provided and not None",
                    config,
                    self,
                )

            # check type (None matches all types)
            if val is not None and not isinstance(val, type_):
                raise InvalidRunConfigException(
                    f"Run option: {cfg_key}, must be of type: {type_.__name__},"
                    f" but was: {val} ({type(val).__name__})",
                    config,
                    self,
                )

            # not required and not set, set to default
            if val is None:
                resolved_cfg.set(cfg_key, default)
        return resolved_cfg

    def __repr__(self):
        # make it a pretty printable dict
        pretty_opts = {}
        for cfg_key, (default, type_, required, help) in self._opts.items():
            key = f"*{cfg_key}" if required else cfg_key
            opt = {"type": type_.__name__}
            if required:
                opt["required"] = True
            else:
                opt["default"] = default
            opt["help"] = help

            pretty_opts[key] = opt
        import pprint

        return pprint.pformat(
            pretty_opts,
            indent=2,
            width=80,
        )


class InvalidRunConfigException(Exception):
    """
    Raised when the supplied ``RunConfig`` does not satisfy the
    ``runopts``, either due to missing required configs or value
    type mismatch.
    """

    def __init__(self, invalid_reason: str, run_config: RunConfig, runopts: "runopts"):
        super().__init__(f"{invalid_reason}. Given: {run_config}, Expected: {runopts}")


class Scheduler(abc.ABC):
    """
    An interface abstracting functionalities of a scheduler.
    Implementors need only implement those methods annotated with
    ``@abc.abstractmethod``.
    """

    def __init__(self, session_name: str):
        self.session_name = session_name

    def submit(self, app: Application, cfg: RunConfig) -> str:
        """
        Submits the application to be run by the scheduler.

        Returns:
            The application id that uniquely identifies the submitted app.
        """
        dryrun_info = self.submit_dryrun(app, cfg)
        return self.schedule(dryrun_info)

    @abc.abstractmethod
    def schedule(self, dryrun_info: AppDryRunInfo) -> str:
        """
        Same as ``submit`` except that it takes an ``AppDryrunInfo``.
        Implementors are encouraged to implement this method rather than
        directly implementing ``submit`` since ``submit`` can be trivially
        implemented by:

        ::

         dryrun_info = self.submit_dryrun(app, cfg)
         return schedule(dryrun_info)

        """

        raise NotImplementedError()

    def submit_dryrun(self, app: Application, cfg: RunConfig) -> AppDryRunInfo:
        """
        Rather than submitting the request to run the app, returns the
        request object that would have been submitted to the underlying
        service. The type of the request object is scheduler dependent.
        This method can be used to dry-run an application. Please refer
        to the scheduler implementation's documentation regarding
        the actual return type.
        """
        resolved_cfg = self.run_opts().resolve(cfg)
        dryrun_info = self._submit_dryrun(app, resolved_cfg)
        dryrun_info._app = app
        dryrun_info._cfg = resolved_cfg
        return dryrun_info

    def _submit_dryrun(self, app: Application, cfg: RunConfig) -> AppDryRunInfo:
        raise NotImplementedError()

    def run_opts(self) -> runopts:
        """
        Returns the run configuration options expected by the scheduler.
        Basically a ``--help`` for the ``run`` API.
        """
        return runopts()

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

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterable:
        """
        Returns an iterator to the log lines of the ``k``th replica of the ``role``.
        The iterator ends end all qualifying log lines have been read.

        If the scheduler supports time-based cursors fetching log lines
        for custom time ranges, then the ``since``, ``until`` fields are
        honored, otherwise they are ignored. Not specifying ``since`` and ``until``
        is equivalent to getting all available log lines. If the ``until`` is
        empty, then the iterator behaves like ``tail -f``, following the log output
        until the job reaches a terminal state.

        The exact definition of what constitutes a log is scheduler specific. Some
        schedulers may consider stderr or stdout as the log, others may read the logs
        from a log file.

        Behaviors and assumptions:

        1. Produces an undefined-behavior if called on an app that does not exist
           The caller should check that the app exists using ``exists(app_id)``
           prior to calling this method.

        2. Is not stateful, calling this method twice with same parameters
           returns a new iterator. Prior iteration
           progress is lost.

        3. Does not always support log-tailing. Not all schedulers support live
           log iteration (e.g. tailing logs while the app is running). Refer to
           the specific scheduler's documentation for the iterator's behavior.

        4. Does not guarantee log retention. It is possible that by the time this
           method is called, the underlying scheduler may have purged the log records
           for this application. If so this method raises an arbitrary exception.

        5. Only raises a ``StopIteration`` exception when the accessible log lines
           have been fully exhausted and the app has reached a final state. For instance,
           if the app gets stuck and does not produce any log lines, then the iterator
           blocks until the app eventually gets killed (either via timeout or manually)
           at which point it raises a ``StopIteration``.

        6. Need not be supported by all schedulers.

        7. Some schedulers may support line cursors by supporting ``__getitem__``
           (e.g. ``iter[50]`` seeks to the 50th log line).

        Returns:
            An ``Iterator`` over log lines of the specified role replica

        Raises:
            NotImplementedError - if the scheduler does not support log iteration
        """
        raise NotImplementedError(
            f"{self.__class__.__qualname__} does not support application log iteration"
        )

    def _validate(self, app: Application, scheduler: SchedulerBackend) -> None:
        """
        Validates whether application is consistent with the scheduler.

        Raises:
            ValueError: if application is not compatible with scheduler
        """
        for role in app.roles:
            if role.container.resources == NULL_RESOURCE:
                raise ValueError(
                    f"No resources for container: {role.container.image}."
                    f" Did you forget to call container.require(resources)"
                )


class MalformedAppHandleException(Exception):
    """
    Raised when APIs are given a bad app handle.
    """

    def __init__(self, app_handle: str):
        super().__init__(
            f"{app_handle} is not of the form: <scheduler_backend>://<session_name>/<app_id>"
        )


class SessionMismatchException(Exception):
    """
    Raised on session certain action APIs
    when the session_name on an app handle does not match
    the current session's name. Modify/update APIs raise
    this exception as modifying/updataing an application
    owned by a different session should not be allowed.
    """

    def __init__(self, app_handle: str, session_name: str):
        super().__init__(
            f"App handle: {app_handle} is not owned by this session: {session_name}."
            f" Please perform the action on the correct session"
            f" or re-run the app on this session"
        )


class UnknownSchedulerException(Exception):
    def __init__(self, scheduler_backend: SchedulerBackend):
        super().__init__(
            f"Scheduler backend: {scheduler_backend} does not exist."
            f" Use session.scheduler_backends() to see all supported schedulers"
        )


# encodes information about a running app in url format
# {scheduler_backend}://{session_name}/{app_id}
AppHandle = str


class UnknownAppException(Exception):
    """
    Raised by ``Session`` APIs when either the application does not
    exist or the application is not owned by the session.
    """

    def __init__(self, app_handle: "AppHandle"):
        super().__init__(
            f"Unknown app = {app_handle}. Did you forget to call session.run()?"
            f" Otherwise, the app may have already finished and purged by the scheduler"
        )


def make_app_handle(
    scheduler_backend: SchedulerBackend, session_name: str, app_id: str
):
    return f"{scheduler_backend}://{session_name}/{app_id}"


def parse_app_handle(app_handle: AppHandle) -> Tuple[SchedulerBackend, str, str]:
    """
    parses the app handle into ```(scheduler_backend, session_name, and app_id)```
    """

    # parse it manually b/c currently tsm does not
    # define allowed characters nor length for session name and app_id
    import re

    pattern = r"(?P<scheduler_backend>.+)://(?P<session_name>.+)/(?P<app_id>.+)"
    match = re.match(pattern, app_handle)
    if not match:
        raise MalformedAppHandleException(app_handle)
    gd = match.groupdict()
    return gd["scheduler_backend"], gd["session_name"], gd["app_id"]


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

    def run(
        self,
        app: Application,
        scheduler: SchedulerBackend = "default",
        cfg: Optional[RunConfig] = None,
    ) -> AppHandle:
        """
        Runs the given application in the specified mode.

        .. note:: sub-classes of ``Session`` should implement ``schedule`` method
                  rather than overriding this method directly.

        Returns:
            An application handle that is used to call other action APIs on the app.

        Raises:
            AppNotReRunnableException: if the session/scheduler does not support re-running attached apps
        """

        dryrun_info = self.dryrun(app, scheduler, cfg)
        return self.schedule(dryrun_info)

    @abc.abstractmethod
    def schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        """
        Actually runs the application from the given dryrun info.
        Useful when one needs to overwrite a parameter in the scheduler
        request that is not configurable from one of the object APIs.

        .. warning:: Use sparingly since abusing this method to overwrite
                     many parameters in the raw scheduler request may
                     lead to your usage of TSM going out of compliance
                     in the long term. This method is intended to
                     unblock the user from experimenting with certain
                     scheduler-specific features in the short term without
                     having to wait until TSM exposes scheduler features
                     in its APIs.

        .. note:: It is recommended that sub-classes of ``Session`` implement
                  this method instead of directly implementing the ``run`` method.

        Usage:

        ::

         dryrun_info = session.dryrun(app, scheduler="default", cfg)

         # overwrite parameter "foo" to "bar"
         dryrun_info.request.foo = "bar"

         app_handle = session.submit(dryrun_info)

        """
        raise NotImplementedError()

    def dryrun(
        self,
        app: Application,
        scheduler: SchedulerBackend = "default",
        cfg: Optional[RunConfig] = None,
    ) -> AppDryRunInfo:
        """
        Dry runs an app on the given scheduler with the provided run configs.
        Does not actually submit the app but rather returns what would have been
        submitted. The returned ``AppDryRunInfo`` is pretty formatted and can
        be printed or logged directly.

        Usage:

        ::

         dryrun_info = session.dryrun(app, scheduler="local", cfg)
         print(dryrun_info)

        """
        # input validation
        if not app.roles:
            raise ValueError(
                f"No roles for app: {app.name}. Did you forget to call app.of(roles..)?"
            )

        for role in app.roles:
            if not role.entrypoint:
                raise ValueError(
                    f"No entrypoint for role: {role.name}."
                    f" Did you forget to call role.runs(entrypoint, args, env)?"
                )
            if role.num_replicas <= 0:
                raise ValueError(
                    f"Non-positive replicas for role: {role.name}."
                    f" Did you forget to call role.replicas(positive_number)?"
                )
            if role.container == NULL_CONTAINER:
                raise ValueError(
                    f"No container for role: {role.name}."
                    f" Did you forget to call role.on(container)"
                )
        dryrun_info = self._dryrun(app, scheduler, cfg or RunConfig())
        dryrun_info._scheduler = scheduler
        return dryrun_info

    @abc.abstractmethod
    def _dryrun(
        self, app: Application, scheduler: SchedulerBackend, cfg: RunConfig
    ) -> AppDryRunInfo:
        """
        The actual dryrun logic.
        Implementors of ``Session`` should implement this method rather than ``dryrun``.
        """
        raise NotImplementedError()

    def run_opts(self) -> Dict[str, runopts]:
        """
        Returns the ``runopts`` for the supported scheduler backends.

        Usage:

        ::

         local_runopts = session.run_opts()["local"]
         print("local scheduler run options: {local_runopts}")

        Returns:
            A map of scheduler backend to its ``runopts``
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def scheduler_backends(self) -> List[SchedulerBackend]:
        """
        Returns a list of all supported scheduler backends.
        All session implementations must support a "default"
        scheduler backend and document what the default
        scheduler is.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        """
        Returns:
            The status of the application, or ``None`` if the app does not exist anymore
            (e.g. was stopped in the past and removed from the scheduler's backend).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def wait(self, app_handle: AppHandle) -> Optional[AppStatus]:
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
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list(self) -> Dict[AppHandle, Application]:
        """
        Returns the applications that were run with this session mapped by the app handle.
        The persistence of the session is implementation dependent.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self, app_handle: AppHandle) -> None:
        """
        Stops the application, effectively directing the scheduler to cancel
        the job. Does nothing if the app does not exist.

        .. note:: This method returns as soon as the cancel request has been
                  submitted to the scheduler. The application will be in a
                  ``RUNNING`` state until the scheduler actually terminates
                  the job. If the scheduler successfully interrupts the job
                  and terminates it the final state will be ``CANCELLED``
                  otherwise it will be ``FAILED``.

        Raises:
            SessionMismatchException: if the app handle does not belong to this session
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def describe(self, app_handle: AppHandle) -> Optional[Application]:
        """
        Reconstructs the application (to the best extent) given the app handle.
        Note that the reconstructed application may not be the complete app as
        it was submitted via the run API. How much of the app can be reconstructed
        is scheduler dependent.

        Returns:
            Application or None if the app does not exist anymore or if the
            scheduler does not support describing the app handle
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterable:
        """
        Returns an iterator over the log lines of the specified job container.

        .. note:: #. ``k`` is the node (host) id NOT the ``rank``.
                  #. ``since`` and ``until`` need not always be honored (depends on scheduler).

        .. warning:: The semantics and guarantees of the returned iterator is highly
                     scheduler dependent. See ``torchelastic.tsm.driver.api.Scheduler.log_iter``
                     for the high-level semantics of this log iterator. For this reason
                     it is HIGHLY DISCOURAGED to use this method for generating output
                     to pass to downstream functions/dependencies. This method
                     DOES NOT guarantee that 100% of the log lines are returned.
                     It is totally valid for this method to return no or partial log lines
                     if the scheduler has already totally or partially purged log records
                     for the application.

        Usage:

        ::

         app_handle = session.run(app, scheduler="local", cfg=RunConfig())

         print("== trainer node 0 logs ==")
         for line in session.log_lines(app_handle, "trainer", k=0):
            print(line)

        Discouraged anti-pattern:

        ::

         # DO NOT DO THIS!
         # parses accuracy metric from log and reports it for this experiment run
         accuracy = -1
         for line in session.log_lines(app_handle, "trainer", k=0):
            if matches_regex(line, "final model_accuracy:[0-9]*"):
                accuracy = parse_accuracy(line)
                break
         report(experiment_name, accuracy)

        Args:
            app_handle: application handle
            role_name: role within the app (e.g. trainer)
            k: k-th replica of the role to fetch the logs for
            regex: optional regex filter, returns all lines if left empty
            since: datetime based start cursor. If left empty begins from the
                    first log line (start of job).
            until: datetime based end cursor. If left empty, follows the log output
                    until the job completes and all log lines have been consumed.

        Returns:
             An iterator over the role k-th replica of the specified application.

        Raise:
            UnknownAppException: if the app does not exist in the scheduler
            SessionMismatchException: if the app handle does not belong to this session

        """
        raise NotImplementedError()
