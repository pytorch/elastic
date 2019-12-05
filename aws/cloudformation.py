#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import getpass
import logging
import os
import random
import string

from jinja2 import Template
from util import wait_for


log = logging.getLogger(__name__)


class CloudFormation:
    def __init__(self, session):
        self._session = session
        self._cfn = session.client("cloudformation")

    def create_specs_file(self, specs_file, s3_bucket_name, efs_id):
        username = getpass.getuser()
        rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        hash = f"{username}-{rand}"
        stack_name = f"torchelastic-{hash}"
        this_dir = os.path.dirname(__file__)
        cfn_template = os.path.join(this_dir, "cfn/setup.yml")
        sample_specs = os.path.join(this_dir, "config/sample_specs.json")

        params = {
            "WorkerRoleName": f"torchelastic_worker_role-{hash}",
            "RendezvousRoleName": f"torchelastic_rendezvous_role-{hash}",
        }

        if s3_bucket_name:
            params["S3BucketName"] = s3_bucket_name
        if efs_id:
            params["EFSFileSystemId"] = efs_id

        self.create_stack(stack_name, cfn_template, **params)

        for _ in wait_for(
            f"cfn stack: {stack_name} to create", timeout=600, interval=2
        ):
            status, outputs = self.describe_stack(stack_name)
            if status == "CREATE_COMPLETE":
                break
            elif status == "CREATE_FAILED" or status.startswith("ROLLBACK_"):
                # when stack creation fails cfn starts rolling the stack back
                raise RuntimeError(
                    f"Error creating stack {stack_name}, status = {status}"
                )

        outputs["User"] = username

        log.info(f"Writing specs file to: {specs_file}")
        with open(sample_specs) as f:
            specs_template = Template(f.read())
            specs_template.stream(**outputs).dump(specs_file)

    def describe_stack(self, stack_name):
        describe_res = self._cfn.describe_stacks(StackName=stack_name)

        stacks = describe_res["Stacks"]
        if len(stacks) > 1:
            raise RuntimeError(f"Found more than one stack with name {stack_name}")

        stack_desc = stacks[0]
        status = stack_desc["StackStatus"]

        # cfn outputs an array of maps, each element in the array is
        # a single output of the form "{OutputKey: <key>, OutputValue: <value>}"
        # simplify to a map of <key>, <value>  pairs
        outputs = {}
        if "Outputs" in stack_desc:
            for cfn_output in stack_desc["Outputs"]:
                key = cfn_output["OutputKey"]
                value = cfn_output["OutputValue"]
                outputs[key] = value
        return status, outputs

    def create_stack(self, stack_name, cfn_template, **params):
        log.info(f"Creating cloudformation stack with template: {cfn_template}")

        with open(cfn_template) as f:
            template_body = f.read()

        cfn_parameters = []
        for key, value in params.items():
            cfn_parameters.append({"ParameterKey": key, "ParameterValue": value})

        res = self._cfn.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_NAMED_IAM"],
            Parameters=cfn_parameters,
        )

        return res["StackId"]
