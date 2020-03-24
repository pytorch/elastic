#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist


class RendezvousParameters(object):

    __slots__ = (
        "rdzv_backend",
        "rdzv_endpoint",
        "run_id",
        "min_workers",
        "max_workers",
        "config",
    )

    def __init__(
        self,
        rdzv_backend: str,
        rdzv_endpoint: str,
        run_id: str,
        min_workers: int,
        max_workers: int,
        config=None,
    ):
        """
        rdzv_backend (str): The rdzv_backend that is used to register
                            rendezvous, e.g. etcd
        rdzv_endpoint (str): The rdzv_endpoint of the rendezvous. Usually it is a string
                        in the format host:port
        run_id (str): The unique job id that is used in rendezvous.
        min_workers (int): The min amount of workers that is required to
                           complete rendezvous.
        max_workers (int): The max amount of workers that can participate in
                           rendezvous.
        config (str): A comma-separated list of additional  configuration
                      of the following format: key1=value1,key2=value2
        """
        self.rdzv_backend = rdzv_backend
        self.rdzv_endpoint = rdzv_endpoint
        self.run_id = run_id
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.config = config if config is not None else ""

        assert 0 < min_workers <= max_workers
        assert rdzv_backend is not None


def _construct_rendezvous_url(params: RendezvousParameters):
    url_suffix = ""
    for kv in params.config.split(","):
        if kv:
            conf_key, conf_val = kv.split("=")
            url_suffix += f"&{conf_key}={conf_val}"

    return (
        f"{params.rdzv_backend}://{params.rdzv_endpoint}/{params.run_id}"
        f"?min_workers={params.min_workers}"
        f"&max_workers={params.max_workers}"
        f"{url_suffix}"
    )


def get_rendezvous(params: RendezvousParameters):
    """
    Given a rdzv parameters, tries to construct a url and find
    the rendezvous handler in using pytorch.distributed.
    The handler should be registerred before this method is invoked.
    The registration of the handlers are done autmatically during
    the module import.
    """
    url = _construct_rendezvous_url(params)
    return dist.rendezvous(url)
