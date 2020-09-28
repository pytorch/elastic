#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Callable, Dict, Tuple


class RendezvousClosedException(Exception):
    """
    Raised when a rendezvous for the specified run_id is closed.
    This is used to signal completion to nodes that arrive late.
    """

    pass


class RendezvousTimeoutException(Exception):
    """
    Raised from ``RendezvousHandler.next_rendezvous()`` to signal that the
    rendezvous did not
    succeed within the allocated time. This is meant to be interpreted
    as a non-retryable type of failure.
    """

    pass


class RendezvousNonRetryableError(Exception):
    """
    Raised from any of the ``RendezvousHandler`` methods when a failure
    occured that should not be retried with the same worker process.
    """

    pass


class RendezvousHandler(abc.ABC):
    """
    Main rendezvous interface.

    .. note:: torchelastic users normally **do not** need to implement their
              own ``RendezvousHandler``. An implementation based on
              `etcd <https://etcd.io/>`__ is already provided, and is recommended
              for most users, provided they can deploy it in their environment.

    .. warning:: torchelastic is currently considered experimental,
                 so the APIs may change!
    """

    @abc.abstractmethod
    def next_rendezvous(
        self,
        # pyre-ignore[11]: Annotation `Store` is not defined as a type.
        # pyre-ignore[10]: Name `torch` is used but not defined.
    ) -> Tuple["torch.distributed.Store", int, int]:  # noqa: F821
        """
        Main entry-point into the rendezvous barrier.
        Blocks until the rendezvous is complete (and the current
        process is included in the formed worker group), or a timeout occurs, or
        rendezvous was marked closed.

        Returns: a tuple of (``c10d Store``, ``rank``, ``world size``)

        Raises:
            RendezvousClosedException - if rendezvous for the current
               job is closed.
            RendezvousTimeoutException - on timeout
        """
        pass

    @abc.abstractmethod
    def is_closed(self) -> bool:
        """
        Checks whether rendezvous for current job has been closed,
        which means all future attempts to re-rendezvous (within same job) will
        fail.

        .. note:: ``is_closed`` and ``set_closed`` have semantics of eventual
                  propagation, and should not be used for synchronization.
                  The intention here is that if at least one worker decides
                  the job is finished, it will close the rendezvous, and
                  other workers will soon observe this and stop
                  training/rendezvous-ing as well.
        """
        pass

    @abc.abstractmethod
    def set_closed(self):
        """
        Used to mark the rendezvous (for current job) as closed.
        """
        pass

    @abc.abstractmethod
    def num_nodes_waiting(self) -> int:
        """
        Returns number of workers who *arrived late* at
        the rendezvous barrier, hence weren’t included in the current worker
        group.

        Callers should periodically call this method to check whether
        new members are waiting to join the job and if so admit them by
        calling ``next_rendezvous()`` (re-rendezvous).
        """
        pass

    @abc.abstractmethod
    def get_run_id(self) -> str:
        """
        Returns the run_id of this rendezvous handler. The run_id is a user-defined
        id that uniquely identifies an instance of a distributed application.
        It typically maps to a job id and is used to allow workers to join the
        correct distributed application.
        """
        pass

    def shutdown(self) -> bool:
        """
        Closes all resources that were open for rendezvous run.

        Usage:

        ::

         def main():
             rdzv_handler = ...
             try:
               rank, world_size, store = rdzv_handler.next_rendezvous()
             finally:
               rdzv_handler.shutdown()
        """
        pass


class RendezvousParameters(object):
    """
    Data object holding necessary and sufficient configuration parameters to
    construct a ``RendezvousHandler`` specified by the ``rdzv_backend``.
    """

    __slots__ = ("backend", "endpoint", "run_id", "min_nodes", "max_nodes", "configs")

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        **kwargs,
    ):
        """
        backend: The rdzv_backend that is used to register rendezvous, e.g. etcd
        endpoint: The rdzv_endpoint of the rendezvous. Usually it is a string
                    in the format host:port
        run_id: The unique job id that is used in rendezvous.
        min_nodes: The min number of nodes required to complete rendezvous.
        max_nodes: The max amount of nodes that are allowed to join the rendezvous.
        **kwargs: Additional configurations for the particular rdzv backend as key, value pairs
        """
        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

        self.configs = {}
        for (key, val) in kwargs.items():
            self.configs[key] = val

        assert 0 < min_nodes <= max_nodes
        assert backend is not None

    def get(self, config_key: str, default_value: Any = None) -> Any:
        """
        Returns the the config value from the configs map
        if one exits or the ``default_value``. Checks for ``None`` values,
        so if the config key exists in ``configs``, but maps to ``None``, then
        the ``default_value`` is returned.

        If the default_value is not specified (or is ``None``) then this method
        interprets the config key as a required config and will raise a ``KeyError``
        if the config key is either not found or maps to ``None``
        """
        if config_key not in self.configs:
            if default_value is None:
                raise KeyError(
                    f"required config: {config_key} not found, and a default was not provided"
                )
            else:
                return default_value

        val = self.configs[config_key]
        if val is None:
            if default_value is not None:
                return default_value
            else:
                raise KeyError(
                    f"required config: {config_key} maps to None, and a default was not provided"
                )
        else:
            return val


class RendezvousHandlerFactory:
    def __init__(self):
        self._factory_method_registry: Dict[
            str, Callable[[RendezvousParameters], RendezvousHandler]
        ] = {}

    def register(
        self,
        rdzv_backend: str,
        factory_method: Callable[[RendezvousParameters], RendezvousHandler],
    ):
        if rdzv_backend in self._factory_method_registry:
            fn = self._factory_method_registry[rdzv_backend]
            raise ValueError(
                f"cannot double register rdzv_backend: {rdzv_backend} "
                f"to {factory_method.__module__}.{factory_method.__name__},"
                f" already registered with: {fn.__module__}.{fn.__name__}"
            )

        self._factory_method_registry[rdzv_backend] = factory_method

    def create_rdzv_handler(self, rdzv_params: RendezvousParameters):
        backend = rdzv_params.backend
        if backend not in self._factory_method_registry:
            raise ValueError(
                f"no factory method found for rdzv_backend: {backend},"
                f" did you forget to call {self.register.__name__}?"
            )

        return self._factory_method_registry[backend](rdzv_params)
