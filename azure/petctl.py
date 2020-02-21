from util import *

# Create a Kubernetes specs and YAML job file based on user inputs
def configure(args):  
    configure_yaml(args)
    configure_json(args)

def setup(args):
    azure_login()
    install_aks_engine()
    deploy_aks_cluster(args)

def upload_storage(args):
    upload_to_azure_blob(args)

def storage_secret(args):
    create_storage_secrets(args)

def docker_secret(args):
    create_docker_image_secret(args)

def run_job(args):
    install_blobfuse_drivers()
    commands = ["kubectl delete -f config/azure-pytorch-elastic.yaml",
                "kubectl apply -f config/azure-pytorch-elastic.yaml",
                "kubectl describe pods",
                "kubectl get pods --selector app=azure-pytorch-elastic"]
    
    run_commands(commands)

def check_status():
    commands = ["kubectl describe pods",
                "kubectl get pods --selector app=azure-pytorch-elastic"]
    
    run_commands(commands)
    
def get_logs():
    run_commands(["kubectl logs --selector app=azure-pytorch-elastic "])
    
def delete_resources():
    commands = ["kubectl config delete-cluster azure-pytorch-elastic", 
                "kubectl delete secret pet-blob-secret",
                "kubectl delete namespace --all",
                "RD /S /Q _output"]
    run_commands(commands)
    print('Deleted all resources, please manually delete the AKS resources from the Azure Portal.')

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
        default = "azure-pytorch-elastic",
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
    parser_configure.add_argument(
        "--master_vm",
        type=str,
        required=False,
        default="Standard_DS1_v2",
        help="Azure VM instance for master node",
    )
    parser_configure.add_argument(
        "--worker_vm",
        type=str,
        required=False,
        default="Standard_NC6s_v3",
        help="Azure VM instance for woker nodes",
    )
    parser_configure.set_defaults(func=configure)
    
     # ---------------------------------- #
     #              UPLOAD STORAGE        #
     # ---------------------------------- #

    parser_upload_storage = subparser.add_parser(
        "upload_storage", help="Upload to Azure Blob storage"
    )

    parser_upload_storage.add_argument(
        "--account_name",
        type=str,
        required=True,
        help="Azure Blob storage Account name",
    )

    parser_upload_storage.add_argument(
        "--container_name",
        type=str,
        required=True,
        help="Azure Blob storage container name",
    )

    parser_upload_storage.add_argument(
        "--sas_token",
        type=str,
        required=True,
        help="Azure Blob storage SAS token",
    )

    parser_upload_storage.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to local files",
    )

    parser_upload_storage.set_defaults(func=upload_storage)
    
     # ---------------------------------- #
     #              SETUP SECRETS         #
     # ---------------------------------- #
    
    parser_storage_secret = subparser.add_parser(
        "storage_secret", help="Generate secret for Azure Blob storage"
    )

    parser_storage_secret.add_argument(
        "--account_name",
        type=str,
        required=True,
        help="Azure Blob storage account name",
    )

    parser_storage_secret.add_argument(
        "--account_key",
        type=str,
        required=True,
        help="Azure Blob storage account key",
    )

    parser_storage_secret.set_defaults(func=storage_secret)

    parser_docker_secret = subparser.add_parser(
        "docker_secret", help="Generate secret for Docker Image"
    )

    parser_docker_secret.add_argument(
        "--server",
        type=str,
        required=True,
        help="Docker server",
    )
    
    parser_docker_secret.add_argument(
        "--username",
        type=str,
        required=True,
        help="Docker username",
    )

    parser_docker_secret.add_argument(
        "--password",
        type=str,
        required=True,
        help="Docker password",
    )

    parser_docker_secret.set_defaults(func=docker_secret)

     # ---------------------------------- #
     #              RUN JOB               #
     # ---------------------------------- #
    
    parser_run_job = subparser.add_parser(
        "run_job", help="Run your training job"
    )
    
    parser_run_job.set_defaults(func=run_job)
    
     # ---------------------------------- #
     #              CHECK STATUS          #
     # ---------------------------------- #
    
    parser_check_status = subparser.add_parser(
        "check_status", help="Check status of your jobs"
    )
    parser_run_job.set_defaults(func=check_status)

     # ---------------------------------- #
     #            DELETE RESOURCES        #
     # ---------------------------------- #
    parser_delete_resources = subparser.add_parser(
        "delete_resources", help="Deletes the kubernetes cluster and all namespaces and secrets"
    )
    parser_delete_resources.set_defaults(func=delete_resources)

     # ---------------------------------- #
     #            GET LOGS                #
     # ---------------------------------- #
    
    parser_get_logs = subparser.add_parser(
        "get_logs", help="Get logs from all your pods"
    )
    
    parser_get_logs.set_defaults(func=get_logs)


     # ---------------------------------- #
     #            SCALE CLUSTER                #
     # ---------------------------------- #
    parser_scale = subparser.add_parser(
        "scale", help="Scale up/down your cluster"
    )
    
    parser_scale.add_argument(
        "--subscription_id",
        type=str,
        required=True,
        help="Subscription id of the cluster",
    )
    
    parser_scale.add_argument(
        "--rg",
        type=str,
        required=True,
        help="Resource group of the cluster",
    )
    
    parser_scale.add_argument(
        "--location",
        type=str,
        required=True,
        help="Location of the cluster",
    )
    
    parser_scale.add_argument(
        "--client_id",
        type=str,
        required=True,
        help="Service principal client id",
    )
    
    parser_scale.add_argument(
        "--client_secret",
        type=str,
        required=True,
        help="Service Principal client secret",
    )
    
    parser_scale.add_argument(
        "--new_node_count",
        type=int,
        required=True,
        help="New node count to scale cluster to",
    )
    
    parser_scale.set_defaults(func=scale_cluster)
    
    args = parser.parse_args()
    
    # -----
    # Execution order: Configure --> Setup --> Run
    # -----
    if args.command == "configure":
        configure(args)
    elif args.command == "setup":
        setup(args)
    elif args.command == "upload_storage":
        upload_storage(args)
    elif args.command == "storage_secret":
        storage_secret(args)
    elif args.command == "docker_secret":
        docker_secret(args)
    elif args.command == "run_job":
        run_job(args)
    elif args.command == "check_status":
        check_status()
    elif args.command == "delete_resources":
        delete_resources()
    elif args.command == "get_logs":
        get_logs()
    elif args.command == "scale":
        scale_cluster(args)
    else:
        print("petctl.py: error: argument command: NULL command: (choose from 'setup', 'configure', 'run_job')")