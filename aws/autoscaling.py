#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from enum import Enum, unique

from jinja2 import Template
from util import wait_for


log = logging.getLogger(__name__)


@unique
class Accelerator(Enum):
    NONE = 0
    GPU = 1

    @classmethod
    def get_accelerator(cls, instance_type):
        """
        get_accelerator("p3.2xlarge") returns Accelerator.GPU
        get_accelerator("i3.xlarge") returns Accelerator.NONE
        """

        instance_accelerators = {
            "g2": Accelerator.GPU,
            "g3": Accelerator.GPU,
            "g4": Accelerator.GPU,
            "p2": Accelerator.GPU,
            "p3": Accelerator.GPU,
        }

        instance_family = instance_type[0:2]
        return instance_accelerators.get(instance_family, Accelerator.NONE)

    @classmethod
    def from_str(cls, accelerator_str):
        """
        returns the enum Accelerator value from a string representation
        """
        accelerators = {"none": Accelerator.NONE, "gpu": Accelerator.GPU}
        return accelerators.get(accelerator_str.lower(), Accelerator.NONE)

    def describe(self):
        """
        Returns a string representation of the enum.
        This method is intended to be used to label certain AWS
        resources in their descriptions/names for informative purposes

        e.g. launch template created for GPUs can be named as: torchelastic_gpu
        """

        string_rep = {Accelerator.NONE.value(): "cpu", Accelerator.GPU.value(): "gpu"}
        return string_rep.get(self, "unknown_accelerator")


class AutoScalingGroup:
    def __init__(self, session):
        self._session = session
        self._asg = session.client("autoscaling")
        self._ec2 = session.client("ec2")

    def get_user_data(self, user_data_template, **kwargs):
        if os.path.isabs(user_data_template):
            user_data_path = user_data_template
        else:
            user_data_path = os.path.join(os.path.dirname(__file__), user_data_template)

        with open(user_data_path) as f:
            user_data_template = Template(f.read())
            user_data = user_data_template.render(**kwargs)
            return user_data

    def get_ami_id(self, accelerator):
        """
        Use EKS optimized AMI since it has everything we need pre-installed
        """

        eks_owner_id = "602401143452"
        eks_amis = {
            Accelerator.NONE: "amazon-eks-node-1.14-v20190927",
            Accelerator.GPU: "amazon-eks-gpu-node-1.14-v20190927",
        }

        res = self._ec2.describe_images(
            Filters=[
                {"Name": "owner-id", "Values": [eks_owner_id]},
                {
                    "Name": "name",
                    "Values": [eks_amis.get(accelerator, Accelerator.NONE)],
                },
            ]
        )
        images = res["Images"]
        assert (
            len(images) == 1
        ), f"Multiple EKS AMIs found for {self._session.aws_region()}"
        return images[0]["ImageId"]

    def create_launch_config(
        self,
        name,
        instance_type,
        instance_role,
        user_data_template,
        security_groups=None,
        accelerator="gpu",
        max_spot_price=None,
        ebs_volume_gb=128,
        **user_data_kwargs,
    ):

        req = {
            "LaunchConfigurationName": name,
            "InstanceType": instance_type,
            "IamInstanceProfile": instance_role,
            "ImageId": self.get_ami_id(Accelerator.from_str(accelerator)),
            "SecurityGroups": security_groups,
            "AssociatePublicIpAddress": True,
            "UserData": self.get_user_data(user_data_template, **user_data_kwargs),
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": ebs_volume_gb,
                        "VolumeType": "gp2",
                        "DeleteOnTermination": True,
                    },
                }
            ],
        }

        if max_spot_price:
            req["SpotMaxPrice"] = str(max_spot_price)

        log.info(f"Creating launch config: {name}")
        self._asg.create_launch_configuration(**req)

    def describe_launch_config(self, name):
        res = self._asg.describe_launch_configurations(LaunchConfigurationNames=[name])
        lcs = res["LaunchConfigurations"]
        return lcs[0] if len(lcs) == 1 else None

    def delete_launch_config(self, name):
        if self.describe_launch_config(name):
            log.info(f"Deleting asg launch config: {name}")
            self._asg.delete_launch_configuration(LaunchConfigurationName=name)

    def create_asg(self, name, size, min_size=None, max_size=None, **kwargs):
        """
        Creates an asg. For specifications on kwargs see config/sample_specs.json
        """

        if not min_size:
            min_size = size

        if not max_size:
            max_size = size

        assert min_size <= size <= max_size

        kwargs["size"] = size
        kwargs["min_size"] = min_size
        kwargs["max_size"] = max_size
        self.create_launch_config(name, **kwargs)

        log.info(f"Creating autoscaling group: {name}")
        self._asg.create_auto_scaling_group(
            AutoScalingGroupName=name,
            LaunchConfigurationName=name,
            VPCZoneIdentifier=",".join(kwargs["subnets"]),
            MinSize=min_size,
            MaxSize=max_size,
            DesiredCapacity=size,
        )

    def create_asg_sync(self, name, size, min_size=None, max_size=None, **kwargs):
        self.create_asg(name, size, min_size, max_size, **kwargs)
        _, hostnames = self.get_hostnames(name, size)
        return hostnames

    def describe_asg(self, name):
        res = self._asg.describe_auto_scaling_groups(AutoScalingGroupNames=[name])
        asgs = res["AutoScalingGroups"]
        num_asgs = len(asgs)

        return asgs[0] if num_asgs == 1 else None

    def delete_asg(self, name):
        if self.describe_asg(name):
            log.info(f"Deleting autoscaling group: {name}")
            self._asg.delete_auto_scaling_group(
                AutoScalingGroupName=name, ForceDelete=True
            )

            for _ in wait_for(f"instances in {name} to terminate"):
                if not self.describe_asg(name):
                    log.info(f"Deleted autoscaling group: {name}")
                    break

        # launch config needs to be deleted after asg
        self.delete_launch_config(name)

    def list_hostnames(self, name):
        return self.get_hostnames(name, 1)

    def get_hostnames(self, name, size):
        """
        Waits until the asg has at least <size> instances in "InService"
        state and returns their public dns names.
        """
        for _ in wait_for(f"autoscaling group: {name} to reach size >= {size}"):
            asg_desc = self.describe_asg(name)
            if not asg_desc:
                return []
            else:
                instances = asg_desc["Instances"]
                ready_instance_ids = [
                    e["InstanceId"]
                    for e in instances
                    if e["LifecycleState"] == "InService"
                ]
                if len(ready_instance_ids) >= size:
                    paginator = self._ec2.get_paginator("describe_instances")

                    hostnames = []
                    instance_ids = []
                    for e in paginator.paginate(InstanceIds=ready_instance_ids):
                        for r in e["Reservations"]:
                            for i in r["Instances"]:
                                hostnames.append(i["PublicDnsName"])
                                instance_ids.append(i["InstanceId"])
                    return instance_ids, hostnames
