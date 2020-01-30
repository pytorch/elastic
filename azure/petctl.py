from util import *

# Create a Kubernetes specs and YAML job file based on user inputs
def configure(args):
    configure_yaml(args)
    configure_json(args)


def setup(args):
    install_aks_engine()
    deploy_aks_cluster(args)
    set_kubeconfig_environment_var()
    install_nvidia_drivers()
    create_storage_secrets(args)
    install_blobfuse_drivers()
    create_docker_image_secret(args)


def run_job():
    commands = ["kubectl delete -f config/azure-pytorch-elastic.yaml",
                "kubectl apply -f config/azure-pytorch-elastic.yaml",
                "kubectl describe pods",
                "kubectl get pods --selector app=azure-pytorch-elastic"]

    set_kubeconfig_environment_var()
    run_commands(commands)


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

    # -----
    # Execution order: Configure --> Setup --> Run
    # -----
    if args.command == "configure":
        configure(args)
    elif args.command == "setup":
        setup(args)
    elif args.command == "run_job":
        run_job()
    else:
        print("petctl.py: error: argument command: NULL command: (choose from 'setup', 'configure', 'run_job')")