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
from cloudformation import CloudFormation
from s3 import S3


log = logging.getLogger(__name__)
PETCTL_DIR = os.path.join(expanduser("~"), ".petctl")
PETCTL_CONFIG_FILE = os.path.join(PETCTL_DIR, "config")
SPECS_FILE = os.path.join(PETCTL_DIR, "specs.json")


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
    # List hosts in job
    # -----------------------------------------
    parser_list_hosts = subparser.add_parser(
        "list_hosts", help="lists InService hosts in the job"
    )
    parser_list_hosts.add_argument(
        dest="job_name", help="name of the job to list the hosts for"
    )
    parser_list_hosts.set_defaults(func=list_hosts)

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

    # -----------------------------------------
    # Setup
    # -----------------------------------------
    parser_setup = subparser.add_parser(
        "setup", help="creates necessary aws resources and outputs a specs file"
    )
    parser_setup.add_argument(
        "--region", default="us-west-2", help="aws region to setup on"
    )
    parser_setup.add_argument(
        "--s3_bucket",
        help="s3 bucket to use for running petctl (if empty, one is created)",
    )
    parser_setup.add_argument(
        "--efs_id", help="efs id to use, if empty, one is created"
    )

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
    worker_specs["user"] = getpass.getuser()

    instance_type = worker_specs["instance_type"]
    script_args_str = worker_specs["args"]

    log.info(
        f"\n------------------------------------------------------------------\n"
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
        s3_prefix = urlparse(s3_dest).path.strip("/")

    log.info(f"Uploading: {script_path} to s3://{s3_bucket}/{s3_prefix}")
    s3 = S3(session)
    url = s3.cp(script_path, s3_bucket, s3_prefix)
    log.info(f"Finished uploading to: {url}")


def list_hosts(session, specs_json, args):
    job_name = args.job_name
    asg = AutoScalingGroup(session)
    asgs = [f"{job_name}_rdzv", f"{job_name}_worker"]
    hosts = {}

    for asg_name in asgs:
        instance_ids, hostnames = asg.list_hostnames(asg_name)
        hosts[asg_name] = zip(instance_ids, hostnames)

    print(f"\n--------------------------------------------------------------")
    for asg_name in hosts:
        print(f"Hosts in {asg_name}:")
        for i, host in enumerate(hosts[asg_name], start=1):
            instance_id = host[0]
            public_dns = host[1]
            print(f"  {i}) {instance_id} ({public_dns})")
        print(f"--------------------------------------------------------------")
    print("To SSH run:")
    print(f"\taws ssm start-session --target <instance_id>")
    print(f"--------------------------------------------------------------")


def configure(args):
    """
    Configures petctl. Writes a simple json config file indicating
    the specs file to use and the aws region to the petctl config directory
    (default ~/.petctl). Prompts the user to input the specs file location
    and aws region.
    """
    while True:
        specs_file = input(
            "Absolute path to specs file (e.g. /home/${USER}/specs.json): "
        )
        if os.path.isfile(specs_file):
            break
        print(f"[{specs_file}] does not exist! Provide an existing path")

    while True:
        region = input("Default aws region to use (e.g. us-west-2): ")
        if region:
            break
        print("AWS region cannot be empty!")

    write_config_file(region, specs_file)
    log.info(f"Configuration complete. petctl config file: {PETCTL_CONFIG_FILE}")


def setup(args):
    """
    Similar to config but creates AWS resources using cfn template
    and based on the cfn stack output, creates the specs file for the user,
    then writes petctl config.
    """
    region = args.region
    s3_bucket_name = args.s3_bucket
    efs_id = args.efs_id
    os.makedirs(PETCTL_DIR, exist_ok=True)
    session = auth.get_session(region)
    cfn = CloudFormation(session)
    cfn.create_specs_file(SPECS_FILE, s3_bucket_name, efs_id)
    write_config_file(region, SPECS_FILE)
    log.info(f"Setup complete. petctl config file: {PETCTL_CONFIG_FILE}")


def write_config_file(region, specs_file):
    petctl_config = {"specs_file": specs_file, "region": region}
    os.makedirs(PETCTL_DIR, exist_ok=True)
    with open(PETCTL_CONFIG_FILE, "w+") as f:
        json.dump(petctl_config, f, indent=4)


def load_configuration():
    if os.path.isfile(PETCTL_CONFIG_FILE):
        with open(PETCTL_CONFIG_FILE) as f:
            return json.load(f)
    else:
        return {}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
    )

    petctl_configs = load_configuration()
    args = parse_arguments(sys.argv, **petctl_configs)

    if args.command == "setup":
        args = parse_arguments(sys.argv)
        setup(args)
    elif args.command == "configure":
        configure(args)
    else:
        log.info(
            f"{PETCTL_CONFIG_FILE} not found or is empty,"
            f" consider running: petctl setup|configure"
        )
        region = args.region
        specs_json = load_specs_json(args.specs_file)
        session = auth.get_session(region)
        args.func(session, specs_json, args)
