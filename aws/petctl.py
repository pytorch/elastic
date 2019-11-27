#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import json
import logging
import os
import sys
from os.path import expanduser
from urllib.parse import urlparse

import auth
from autoscaling import AutoScalingGroup
from s3 import S3


log = logging.getLogger(__name__)
PETCTL_CONFIG = os.path.join(expanduser("~"), ".petctl/config")


def split_args(args, delimiter="--"):
    if delimiter in args:
        idx = args.index(delimiter)
        if idx == (len(args) - 1):
            return args, []
        else:
            return args[0:idx], args[idx + 1 :]
    else:
        return args, []


def parse_arguments(args, **default_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", help="the aws region to operate on")
    parser.add_argument(
        "--specs_file",
        help="see https://github.com/pytorch/elastic/blob/master/aws/README.md#create-specs-file",  # noqa B950
    )
    parser.set_defaults(**default_args)

    subparser = parser.add_subparsers(
        title="actions", description="run_job | kill_job", dest="command"
    )

    # -----------------------------------------
    # Run Job
    # -----------------------------------------
    parser_run_job = subparser.add_parser(
        "run_job", help="runs a torchelastic job on asg"
    )
    parser_run_job.add_argument("--name", required=True, help="name of the job")
    parser_run_job.add_argument(
        "--min_size",
        type=int,
        required=False,
        help="minimum number of worker hosts to continue training",
    )
    parser_run_job.add_argument(
        "--max_size",
        type=int,
        required=False,
        help="maximum number of worker hosts to allow scaling out",
    )
    parser_run_job.add_argument(
        "--size",
        type=int,
        required=True,
        help="number  of worker hosts to start the job with",
    )
    parser_run_job.add_argument(
        "--instance_type", required=False, help="Instance type to run the job on"
    )
    parser_run_job.add_argument(
        dest="script_path",
        help="script or script dir path (e.g. ~/script.py, s3://..., docker://)",
    )
    parser_run_job.set_defaults(func=run_job)

    # -----------------------------------------
    # Kill Job
    # -----------------------------------------
    parser_kill_job = subparser.add_parser(
        "kill_job", help="kills a torchelastic job on asg"
    )

    parser_kill_job.add_argument(dest="job_name", help="name of the job to kill")

    parser_kill_job.set_defaults(func=kill_job)

    # -----------------------------------------
    # Upload script
    # -----------------------------------------
    parser_upload = subparser.add_parser("upload", help="uploads the file/dir to s3")
    parser_upload.add_argument(
        dest="script_path",
        help="script or script dir path (e.g. ~/script.py, s3://..., docker://)",
    )
    parser_upload.add_argument(
        dest="s3_dest",
        help="s3 destination (default: s3://{s3_bucket}/{s3_prefix}/{USER}/scripts)",
    )
    parser_upload.set_defaults(func=upload_script)

    # -----------------------------------------
    # Configure
    # -----------------------------------------
    subparser.add_parser("configure", help="configures petctl")

    petctl_args, script_args = split_args(args[1:])
    parsed = parser.parse_args(petctl_args)
    parsed.script_args = script_args
    return parsed


def load_specs_json(file):
    log.info(f"Loading launch specs from: {args.specs_file}")
    with open(file) as f:
        return json.load(f)


def run_job(session, specs_json, args):
    job_name = args.name
    script_args = args.script_args

    rdzv_specs = specs_json["rdzv"]
    worker_specs = specs_json["worker"]

    script_url = urlparse(args.script_path)
    scheme = script_url.scheme
    if scheme == "docker":
        # docker://tmp/script.py -> tmp/script.py (relative to working dir in docker)
        # docker:///tmp/script.py -> /tmp/script.py (absolute path in docker)
        script = script_url.netloc + script_url.path
    elif scheme == "s3":
        # fetch_and_run supports s3:// so just pass through
        script = args.script_path
    else:
        s3_bucket = worker_specs["s3_bucket"]
        s3_prefix = worker_specs["s3_prefix"]
        script = S3(session).cp(args.script_path, s3_bucket, f"{s3_prefix}/{job_name}")

    asg = AutoScalingGroup(session)
    rdzv_asg_name = f"{job_name}_rdzv"
    worker_asg_name = f"{job_name}_worker"

    # create a single node asg to host the etcd server for rendezvous
    etcd_server_hostname = asg.create_asg_sync(rdzv_asg_name, size=1, **rdzv_specs)[0]
    rdzv_endpoint = f"{etcd_server_hostname}:2379"

    # allow overriding instance types from cli
    if args.instance_type:
        worker_specs["instance_type"] = args.instance_type
    worker_specs["rdzv_endpoint"] = rdzv_endpoint
    worker_specs["job_name"] = job_name
    worker_specs["script"] = script
    worker_specs["args"] = " ".join(script_args)

    instance_type = worker_specs["instance_type"]
    script_args_str = worker_specs["args"]

    log.info(
        f"------------------------------------------------------------------\n"
        f"Starting job...\n"
        f"  job name     : {job_name}\n"
        f"  instance type: {instance_type}\n"
        f"  size         : {args.size} (min={args.min_size}, max={args.max_size})\n"
        f"  rdzv endpoint: {rdzv_endpoint}\n"
        f"  cmd          : {script}\n"
        f"  cmd args     : {script_args_str}\n"
        f"------------------------------------------------------------------\n"
    )

    asg.create_asg(
        worker_asg_name, args.size, args.min_size, args.max_size, **worker_specs
    )


def kill_job(session, specs_json, args):
    job_name = args.job_name
    log.info(f"Killing job {job_name}")
    asg = AutoScalingGroup(session)
    asg.delete_asg(f"{job_name}_rdzv")
    asg.delete_asg(f"{job_name}_worker")


def upload_script(session, specs_json, args):
    script_path = args.script_path
    s3_dest = args.s3_dest

    if not s3_dest:
        s3_bucket = specs_json["s3_bucket"]
        s3_prefix = os.path.join(specs_json["s3_prefix"], getpass.getuser())
    else:
        s3_bucket = urlparse(s3_dest).netloc
        s3_prefix = urlparse(s3_dest).path

    log.info(f"Uploading: {script_path} to s3://{s3_bucket}/{s3_prefix}")
    s3 = S3(session)
    url = s3.cp(script_path, s3_bucket, s3_prefix)
    log.info(f"Finished uploading to: {url}")


def configure():
    specs_file = input("Absolute path to specs file (e.g. /home/${USER}/specs.json): ")
    region = input("Default aws region to use (e.g. us-west-2): ")

    petctl_config = {"specs_file": specs_file, "region": region}
    os.makedirs(os.path.dirname(PETCTL_CONFIG), exist_ok=True)
    with open(PETCTL_CONFIG, "w+") as f:
        json.dump(petctl_config, f, indent=4)

    log.info(f"Configuration complete. petctl config file: {PETCTL_CONFIG}")


def load_configuration():
    if os.path.isfile(PETCTL_CONFIG):
        with open(PETCTL_CONFIG) as f:
            return json.load(f)
    else:
        log.warning(f"{PETCTL_CONFIG} not found, consider running: petctl configure")
        return {}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
    )

    args = parse_arguments(sys.argv, **load_configuration())

    if args.command == "configure":
        configure()
    else:
        region = args.region
        specs_json = load_specs_json(args.specs_file)
        session = auth.get_session(region)
        args.func(session, specs_json, args)
