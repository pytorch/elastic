import argparse
import os
import subprocess
import yaml
import json
import uuid


def configure_yaml(yaml_file, petctl_dir, args):
    print('Configuring azure-pytorch-elastic.yaml....')

    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    data["spec"]["parallelism"] = args.max_size
    data["spec"]["template"]["spec"]["containers"][0]["env"].extend(
        [dict([('name', 'JOB_ID'), ('value', str(uuid.uuid1()) + "_" + args.name)]),
         dict([('name', 'MIN_SIZE'), ('value', str(args.min_size))]),
         dict([('name', 'MAX_SIZE'), ('value', str(args.max_size))])])
    result_yaml = open(os.path.join(petctl_dir, "azure-pytorch-elastic.yaml"), "w")
    yaml.dump(data, result_yaml)


def configure_json(json_file, petctl_dir, args):
    print('Configuring kubernetes.json file...')

    with open(json_file) as f:
        data = json.load(f)
    data["properties"]["masterProfile"]["count"] = 1
    data["properties"]["agentPoolProfiles"][0]["count"] = args.max_size

    result_json = open(os.path.join(petctl_dir, "kubernetes.json"), "w")
    json.dump(data, result_json, indent=4)


# Create a YAML job file based on user inputs
def configure(args):
    PETCTL_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), 'config/'))
    SAMPLE_YAML_FILE = os.path.join(PETCTL_DIR, "sample_specs.yaml")
    KUBERNETES_JSON_FILE = os.path.join(PETCTL_DIR, "kubernetes.json")

    configure_yaml(SAMPLE_YAML_FILE, PETCTL_DIR, args)
    configure_json(KUBERNETES_JSON_FILE, PETCTL_DIR, args)


def run_job():
    subprocess.call("./run_job.sh", shell=True)


def setup(args):
    subprocess.call("./setup.sh " +
                    args.subscription_id + " " +
                    args.dns_prefix + " " +
                    args.rg + " " +
                    args.location + " " +
                    args.client_id + " " +
                    args.client_secret + " " +
                    args.storage_account_name + " " +
                    args.storage_account_key + " " +
                    args.docker_server + " " +
                    args.docker_username + " " +
                    args.docker_password, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(
        title="actions", description="setup | configure | run job", dest="command"
    )

    # ---------------------------------- #
    #               SETUP                #
    # ---------------------------------- #

    parser_setup = subparser.add_parser(
        "setup", help="set up aks-engine, cluster and other dependencies"
    )

    parser_setup.add_argument(
        "--dns_prefix",
        type=str,
        required=False,
        default="azure-pytorch-elastic",
        help="Dns prefix of the app",
    )

    parser_setup.add_argument(
        "--subscription_id",
        type=str,
        required=True,
        help="Subscription id of the cluster",
    )

    parser_setup.add_argument(
        "--rg",
        type=str,
        required=True,
        help="Resource group of the cluster",
    )

    parser_setup.add_argument(
        "--location",
        type=str,
        required=True,
        help="Location of the cluster",
    )

    parser_setup.add_argument(
        "--client_id",
        type=str,
        required=True,
        help="Service principal client id",
    )

    parser_setup.add_argument(
        "--client_secret",
        type=str,
        required=True,
        help="Service Principal client secret",
    )

    parser_setup.add_argument(
        "--storage_account_name",
        type=str,
        required=True,
        help="Pet blob storage secrets",
    )

    parser_setup.add_argument(
        "--storage_account_key",
        type=str,
        required=True,
        help="Pet blob storage secrets",
    )

    parser_setup.add_argument(
        "--docker_server",
        type=str,
        required=True,
        help="Docker server",
    )

    parser_setup.add_argument(
        "--docker_username",
        type=str,
        required=True,
        help="Docker username",
    )

    parser_setup.add_argument(
        "--docker_password",
        type=str,
        required=True,
        help="Docker password",
    )

    parser_setup.set_defaults(func=setup)

    # ---------------------------------- #
    #        CONFIGURE JOB YAML          #
    # ---------------------------------- #

    parser_configure = subparser.add_parser(
        "configure", help="Generate yaml job file"
    )

    parser_configure.add_argument("--name", required=True, help="config parameters")
    parser_configure.add_argument(
        "--min_size",
        type=int,
        required=False,
        help="minimum number of worker hosts to continue training",
    )
    parser_configure.add_argument(
        "--max_size",
        type=int,
        required=False,
        help="maximum number of worker hosts to allow scaling out",
    )
    parser_configure.add_argument(
        "--size",
        type=int,
        required=False,
        help="set size to automatically set min_size = max_size = size",
    )
    parser_configure.set_defaults(func=configure)

    # ---------------------------------- #
    #              RUN JOB               #
    # ---------------------------------- #

    parser_run_job = subparser.add_parser(
        "run_job", help="Run your training job"
    )

    parser_run_job.set_defaults(func=run_job)

    # -----------------------------------------

    args = parser.parse_args()

    if args.command == "setup":
        setup(args)
    elif args.command == "configure":
        configure(args)
    elif args.command == "run_job":
        run_job()
    else:
        print("petctl.py: error: argument command: NULL command: (choose from 'setup', 'configure', 'run_job')")