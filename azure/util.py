#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os
import os.path
import shutil
import subprocess
import tarfile
import textwrap
import urllib.request
import uuid
import zipfile
from os import walk
from shutil import copyfile

import yaml


PETCTL_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Format a multiline command into a single line by trimming white spaces
# and replacing newlines with spaces
def format_command(cmd):
    return textwrap.dedent(cmd).strip().replace(os.linesep, " ")


# This method runs all commands in a separate
# process and returns the output
def run_commands(cmds):
    set_kubeconfig_environment_var()

    for cmd in cmds:
        process = subprocess.run(
            cmd,
            shell=True,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        if process.stdout:
            logger.info(process.stdout)
        if process.stderr:
            logger.info(process.stderr)

    return process.stdout


# Configures job yaml file based on user inputs
def configure_yaml(args):
    SAMPLE_YAML_FILE = os.path.join(PETCTL_DIR, "config", "sample_specs.yaml")
    result_yaml_file = os.path.join(PETCTL_DIR, "config", "azure-pytorch-elastic.yaml")

    logger.info(f"Configuring job yaml {result_yaml_file}")

    with open(SAMPLE_YAML_FILE) as f:
        data = yaml.load(f)

    data["spec"]["parallelism"] = args.max_size
    data["spec"]["template"]["spec"]["containers"][0]["env"].extend(
        [
            {"name": "JOB_ID", "value": str(uuid.uuid1()) + "_" + args.name},
            {"name": "MIN_SIZE", "value": str(args.min_size)},
            {"name": "MAX_SIZE", "value": str(args.max_size)},
        ]
    )

    yaml.dump(data, open(result_yaml_file, "w"))


# Configures job yaml file based on user docker image
def configure_yaml_storage(container_name):
    yaml_file = os.path.join(PETCTL_DIR, "config/azure-pytorch-elastic.yaml")

    logger.info(f"Configuring job yaml {yaml_file}")

    with open(yaml_file) as f:
        data = yaml.load(f)

    data["spec"]["template"]["spec"]["volumes"][0]["flexVolume"]["options"][
        "container"
    ] = container_name

    yaml.dump(data, open(yaml_file, "w"))


# Configures job yaml file based on user docker image
def configure_yaml_docker(image_name):
    yaml_file = os.path.join(PETCTL_DIR, "config/azure-pytorch-elastic.yaml")

    logger.info(f"Configuring job yaml {yaml_file}")

    with open(yaml_file) as f:
        data = yaml.load(f)

    data["spec"]["template"]["spec"]["containers"][0]["image"] = image_name

    yaml.dump(data, open(yaml_file, "w"))


# Configures kubernetes json file based on user inputs
def configure_json(args):
    KUBERNETES_JSON_FILE = os.path.join(PETCTL_DIR, "config/kubernetes.json")
    result_json_file = os.path.join(PETCTL_DIR, "config/", "kubernetes.json")

    logger.info(f"Configuring kubernetes specs {result_json_file}")

    with open(KUBERNETES_JSON_FILE) as f:
        data = json.load(f)
    data["properties"]["masterProfile"]["count"] = 1
    data["properties"]["agentPoolProfiles"][0]["count"] = args.min_size
    data["properties"]["masterProfile"]["vmSize"] = args.master_vm
    data["properties"]["agentPoolProfiles"][0]["vmSize"] = args.worker_vm

    json.dump(data, open(result_json_file, "w"), indent=4)


# Download AKS engine installer script for Linux
def download_aks_engine_script():
    url = (
        "https://raw.githubusercontent.com/Azure/aks-engine/master/scripts/get-akse.sh"
    )
    urllib.request.urlretrieve(url, "config/get-akse.sh")
    logger.info("Downloading aks engine script.....")


# Download AKS engine binary for Windows
def download_aks_engine_script_for_windows():
    print("Downloading aks engine binary.....")
    url = (
        "https://github.com/Azure/aks-engine/releases"
        "/download/v0.47.0/aks-engine-v0.47.0-windows-amd64.zip"
    )
    filename, _ = urllib.request.urlretrieve(url, "config/aks.zip")
    zip_file_object = zipfile.ZipFile(filename, "r")
    for name in zip_file_object.namelist():
        if "aks-engine.exe" in name:
            zip_file_object.extract(name, "aks-engine")
            copyfile("aks-engine/" + name, "aks-engine.exe")
            break


# Installs AKS engine from the script/binary
def install_aks_engine():
    if os.name == "nt":
        download_aks_engine_script_for_windows()
    else:
        download_aks_engine_script()
        commands = ["chmod 700 config/get-akse.sh", "./config/get-akse.sh"]
        run_commands(commands)


# Download AzCopy script to upload to AzureBlobStorage
def download_azcopy_script():
    print("Downloading azcopy cli")
    url = "https://aka.ms/downloadazcopy-v10-linux"
    filename, _ = urllib.request.urlretrieve(url, "config/azcopy.tar.gz")
    tar_file_object = tarfile.open(filename, "r:gz")
    for member in tar_file_object.getmembers():
        if member.isreg():
            member.name = os.path.basename(member.name)
            if "azcopy" == member.name:
                tar_file_object.extract(member.name, ".")
                break


# Download AzCopy script for windows
def download_azcopy_script_for_windows():
    url = "https://aka.ms/downloadazcopy-v10-windows"
    filename, _ = urllib.request.urlretrieve(url, "config/azcopy.zip")
    zip_file_object = zipfile.ZipFile(filename, "r")
    for member in zip_file_object.infolist():
        if not member.is_dir():
            member.filename = os.path.basename(member.filename)
            if "azcopy" in member.filename:
                zip_file_object.extract(member, ".")
                break


"""
 Helper function to upload to AzureBlob storage based on
 Storage account,
 Storage container,
 SAS Token
"""


def upload_to_azure_blob(args):
    destination = (
        f"https://{args.account_name}.blob.core.windows.net/"
        "{args.container_name}{args.sas_token}"
    )

    if os.name == "nt":
        download_azcopy_script_for_windows()
        commands = [
            format_command(
                f"""
                 azcopy copy "{args.source_path}"
                 "{destination}"
                 --recursive=True"""
            )
        ]
        run_commands(commands)
    else:
        download_azcopy_script()
        commands = [
            format_command(
                f"""
                 ./azcopy copy "{args.source_path}"
                 "{destination}"
                 --recursive=True"""
            )
        ]
        run_commands(commands)
    configure_yaml_storage(args.container_name)


"""
 Sets KUBECONFIG environment variable to
 the path to the  json file generated
"""


def set_kubeconfig_environment_var():
    if os.path.isdir("_output"):
        config_path = os.path.join(
            PETCTL_DIR, "_output", "azure-pytorch-elastic", "kubeconfig"
        )
        logger.info(f"Reading KUBECONFIG environment variable from {config_path}")

        for files in walk(config_path):
            for f in files:
                if f and f[0].endswith(".json"):
                    config_path = os.path.join(config_path, f[0])

        if config_path.endswith(".json"):
            os.environ["KUBECONFIG"] = config_path
            logger.info(
                f"Setting KUBECONFIG env variable {os.environ.get('KUBECONFIG')}"
            )


# Create storage secret named 'pet-blob-secret'
def create_storage_secrets(args):
    commands = [
        format_command(
            f"""
             kubectl create secret generic pet-blob-secret
             --from-literal accountname={args.account_name}
             --from-literal accountkey={args.account_key}
             --type=azure/blobfuse"""
        )
    ]
    run_commands(commands)


# Install Azure blobfuse drivers
def install_blobfuse_drivers():
    commands = [
        "kubectl apply -f "
        + "https://raw.githubusercontent.com/Azure/kubernetes-volume-drivers"
        + "/master/flexvolume/blobfuse/deployment/blobfuse-flexvol-installer-1.9.yaml"
    ]
    run_commands(commands)


# Create docker image secrets given user inputs
def create_docker_image_secret(args):
    configure_yaml_docker(args.image_name)
    commands = [
        format_command(
            f"""
             kubectl create secret
             docker-registry pet-docker-secret
             --docker-server={args.server}
             --docker-username={args.username}
             --docker-password={args.password}
             --docker-email='test@test.com'"""
        )
    ]
    run_commands(commands)
    logger.info("Docker image registered..")


# Deploy AKS cluster
def deploy_aks_cluster(args):
    logger.info("Started AKS cluster deployment. This will take some time .....")
    commands = [
        format_command(
            f"""
             aks-engine deploy -f
             --subscription-id {args.subscription_id}
             --dns-prefix {args.dns_prefix}
             --resource-group {args.rg}
             --location {args.location}
             --api-model config/kubernetes.json
             --client-id {args.client_id}
             --client-secret {args.client_secret}
             --set servicePrincipalProfile.clientId={args.client_id}
             --set servicePrincipalProfile.secret={args.client_secret}"""
        )
    ]
    run_commands(commands)


# Scale the cluster up and down based on user input
def scale_cluster(args):
    command = [
        format_command(
            f"""
             aks-engine scale
             --subscription-id {args.subscription_id}
             --resource-group {args.rg}
             --client-id {args.client_id}
             --client-secret {args.client_secret}
             --location {args.location}
             --api-model _output/azure-pytorch-elastic/apimodel.json
             --new-node-count {args.new_node_count}
             --apiserver azure-pytorch-elastic.{4}.cloudapp.azure.com"""
        )
    ]
    run_commands(command)


def delete_resources_util():
    commands = [
        "kubectl config delete-cluster azure-pytorch-elastic",
        "kubectl delete secret pet-blob-secret",
        "kubectl delete namespace --all",
    ]
    run_commands(commands)

    if os.path.isdir("_output"):
        shutil.rmtree(os.path.join(PETCTL_DIR, "_output"))

    logger.info(
        (
            "Deleted all resources,"
            "please manually delete the AKS resources from the Azure Portal."
        )
    )
