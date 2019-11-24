#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile as tar
import tempfile

from . import auth
from .autoscaling import AutoScalingGroup


log = logging.getLogger(__name__)


def split_args(args, delimiter="--"):
    if delimiter in args:
        idx = args.index(delimiter)
        if idx == (len(args) - 1):
            return args, []
        else:
            return args[0:idx], args[idx + 1 :]
    else:
        return args, []


def parse_arguments(args=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        dest="region",
        required=False,
        default="us-west-2",
        help="the aws region to operate on",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enables DEBUG level logging",
    )
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
        "--specs_file",
        type=argparse.FileType("r"),
        required=True,
        help="json file containing static configuration parameters (see config/)",
    )
    parser_run_job.add_argument(
        "--no_upload",
        action="store_true",
        help="does not upload the script (uses script in docker or already in s3)",
    )
    parser_run_job.add_argument(
        dest="script_path",
        help="script or script dir path (e.g. $HOME/workspace/my_script.py)",
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

    petctl_args, script_args = split_args(args[1:])
    parsed = parser.parse_args(petctl_args)
    parsed.script_args = script_args
    return parsed


def load_specs_json(file):
    log.info(f"Loading launch specs from: {args.specs_file}")
    with open(file) as f:
        return json.load(f)


def s3_cp(session, target_path, bucket, key):
    """
    Uploads target_path to s3://bucket/key. If the target_path is a file
    then uploads to s3://bucket/key/file_name, if the target_path is a
    directory, then a tarball is created with the contents of target_path
    and uploaded to s3://bucket/key/dir_name.tar.gz. The tar is created as
    if created by running the command:

    cd target_path && tar xzf /tmp/$(basename target_path).tar.gz *

    Returns the destination s3 url
    """

    target_basename = os.path.basename(target_path)

    if os.path.isdir(target_path):
        tmpdir = tempfile.mkdtemp(prefix="petctl_")
        tar_basename = f"{target_basename}.tar.gz"
        tar_file = os.path.join(tmpdir, tar_basename)
        log.info(f"Compressing {target_path} into {tar_basename}")
        with tar.open(tar_file, "x:gz") as f:
            f.add(target_path, arcname="", recursive=True)

        dest_key = f"{key}/{tar_basename}"
        target_file = tar_file
    else:
        tmpdir = None
        dest_key = f"{key}/{target_basename}"
        target_file = target_path

    log.info(f"Uploading {target_file} to s3://{bucket}/{dest_key}")
    session.client("s3").upload_file(target_file, bucket, dest_key)

    if tmpdir:
        log.info(f"Deleting tmp dir: {tmpdir}")
        shutil.rmtree(tmpdir)
    return f"s3://{bucket}/{dest_key}"


def run_job(session, args):
    job_name = args.name
    script_args = args.script_args

    # TODO make specs into a proper config object?
    specs_json = json.load(args.specs_file)
    rdzv_specs = specs_json["rdzv"]
    worker_specs = specs_json["worker"]

    if args.no_upload:
        # script_path is just passed through, useful for running docker-local
        # scripts or scripts that already have been uploaded to s3
        script = args.script_path
    else:
        s3_bucket = worker_specs["s3_bucket"]
        s3_prefix = worker_specs["s3_prefix"]
        script = s3_cp(session, args.script_path, s3_bucket, f"{s3_prefix}/{job_name}")

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

    AutoScalingGroup(session).create_asg(
        worker_asg_name, args.size, args.min_size, args.max_size, **worker_specs
    )


def kill_job(session, args):
    job_name = args.job_name
    log.info(f"Killing job {job_name}")
    asg = AutoScalingGroup(session)
    asg.delete_asg(f"{job_name}_rdzv")
    asg.delete_asg(f"{job_name}_worker")


if __name__ == "__main__":

    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
    )

    region = args.region
    session = auth.get_session(region)

    args.func(session, args)
