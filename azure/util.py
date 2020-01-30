import yaml
import json
import argparse
import os
import subprocess
import uuid
import urllib.request
PETCTL_DIR  = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
def run_commands(cmds):
    for cmd in cmds:
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, env = os.environ)
        for line in process.stdout:
            print(line)
def configure_yaml(args):
    SAMPLE_YAML_FILE = os.path.join(PETCTL_DIR, "config/sample_specs.yaml")
    result_yaml_file = os.path.join(PETCTL_DIR, 'config/', "azure-pytorch-elastic.yaml")
    
    print('Configuring job yaml ', result_yaml_file)
    
    with open(SAMPLE_YAML_FILE) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    data["spec"]["parallelism"] = args.max_size
    data["spec"]["template"]["spec"]["containers"][0]["env"].extend([dict([('name', 'JOB_ID'), ('value', str(uuid.uuid1()) + "_" + args.name)]),
                                                                     dict([('name', 'MIN_SIZE'), ('value', str(args.min_size))]),
                                                                     dict([('name', 'MAX_SIZE'), ('value', str(args.max_size))])])
    
    yaml.dump(data, open(result_yaml_file,"w"))
def configure_json(args):
    KUBERNETES_JSON_FILE = os.path.join(PETCTL_DIR, "config/kubernetes.json")
    result_json_file = os.path.join(PETCTL_DIR,"config/","kubernetes.json")
    
    print('Configuring kubernetes specs ',result_json_file)
    
    with open(KUBERNETES_JSON_FILE) as f:
        data =  json.load(f)
    data["properties"]["masterProfile"]["count"] = 1
    data["properties"]["agentPoolProfiles"][0]["count"] = args.max_size
    
    json.dump(data, open(result_json_file,"w"), indent=4)
def azure_login():
    run_commands(["az login"])

def download_aks_engine_script():
    url = 'https://raw.githubusercontent.com/Azure/aks-engine/master/scripts/get-akse.sh'
    urllib.request.urlretrieve(url, 'config/get-akse.sh')
    print('Downloading aks engine script.....')
def install_aks_engine():
    download_aks_engine_script()
    commands = ["chmod 700 config/get-akse.sh", "./config/get-akse.sh"]
    run_commands(commands)
def set_kubeconfig_environment_var():
    os.environ["KUBECONFIG"] = PETCTL_DIR + "/_output/aagarg-pytorch-elastic/kubeconfig/kubeconfig.westeurope.json"
    print("Setting KUBECONFIG env variable ", os.environ.get("KUBECONFIG"))
def install_nvidia_drivers():
    commands = ["kubectl create namespace gpu-resources"]
    run_commands(commands)
def create_storage_secrets(args):
    commands = ["kubectl create secret generic pet-blob-secret \
                 --from-literal accountname={0} \
                 --from-literal accountkey='{1}' \
                 --type='azure/blobfuse'"
                 .format(args.storage_account_name,
                         args.storage_account_key)]
    run_commands(commands)
def install_blobfuse_drivers():
    commands = ["kubectl apply -f https://raw.githubusercontent.com/Azure/kubernetes-volume-drivers/master/flexvolume/blobfuse/deployment/blobfuse-flexvol-installer-1.9.yaml"]
    run_commands(commands)
def create_docker_image_secret(args):
    commands = ["kubectl create secret \
                docker-registry pet-docker-secret \
                --docker-server= {0} \
                --docker-username= {1} \
                --docker-password= {2}"
               .format(args.docker_server,
                       args.docker_username,
                    args.docker_password)]
    print("Docker image registered..")
 
 
def deploy_aks_cluster(args):
    commands = ["aks-engine deploy  --subscription-id {0} \
                                   --dns-prefix {1} \
                                   --resource-group {2} \
                                   --location {3} \
                                   --api-model config/kubernetes.json \
                                   --client-id {4} \
                                   --client-secret {5} \
                                   --set servicePrincipalProfile.clientId={4} \
                                   --set servicePrincipalProfile.secret={5}"
                                    .format(args.subscription_id,
                                            args.dns_prefix,
                                            args.rg,
                                            args.location,
                                            args.client_id,
                                            args.client_secret,
                                            )]
    run_commands(commands)