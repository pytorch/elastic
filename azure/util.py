import yaml
import json
import argparse
import os
import os.path
import subprocess
import uuid
import urllib.request

PETCTL_DIR  = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def run_commands(cmds):
    output = []
    set_kubeconfig_environment_var()
    
    for cmd in cmds:
        print("Running {}".format(cmd))
        process = subprocess.Popen(cmd, universal_newlines=True, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, env = os.environ)
        for line in process.stdout:
            print(line)
            output.append(line)
        for err in process.stderr:
            print(err)
    return output

def configure_yaml(args):
    SAMPLE_YAML_FILE       = os.path.join(PETCTL_DIR, "config/sample_specs.yaml")
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
    data["properties"]["agentPoolProfiles"][0]["count"] = args.min_size
    data["properties"]["masterProfile"]["vmSize"] = args.master_vm
    data["properties"]["agentPoolProfiles"][0]["vmSize"] = args.worker_vm
    
    json.dump(data, open(result_json_file,"w"), indent=4)

def azure_login():
    check_cmd = "az account show"
    p = subprocess.Popen(check_cmd, shell=True, stdout=subprocess.PIPE, env = os.environ)
    acctS,_ = p.communicate()
    if (acctS == b''):
        login_cmd = "az login"
        process = subprocess.Popen(login_cmd, shell=True, stdout=subprocess.PIPE, env = os.environ)
        for line in process.stdout:
            print(line)

def download_aks_engine_script():
    url = 'https://raw.githubusercontent.com/Azure/aks-engine/master/scripts/get-akse.sh'
    urllib.request.urlretrieve(url, 'config/get-akse.sh')
    print('Downloading aks engine script.....')

def install_aks_engine():
    download_aks_engine_script()
    commands = ["chmod 700 config/get-akse.sh", "./config/get-akse.sh"]
    run_commands(commands)

def set_kubeconfig_environment_var():
    config_path = PETCTL_DIR + "/_output/azure-pytorch-elastic/kubeconfig/kubeconfig.*.json"
    ls_cmd = "ls {}".format(config_path)
    print("Reading KUBECONFIG environment variable from {}".format(config_path))

    p = subprocess.Popen(ls_cmd, shell=True, stdout=subprocess.PIPE, env = os.environ)
    configS,_ = p.communicate()
    if (configS != b''):
        config_file =  configS.decode('utf-8').rstrip()
        if(os.path.isfile(config_file)): 
            os.environ["KUBECONFIG"] = config_file
            print("Setting KUBECONFIG env variable ", os.environ.get("KUBECONFIG"))
 
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
    run_commands(commands)
    print("Docker image registered..") 
 
def deploy_aks_cluster(args):
    commands = ["aks-engine deploy -f  --subscription-id {0} \
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

def scale_cluster(args):
    command = ["aks-engine scale \
            --subscription-id {0} \
            --resource-group {1} \
            --client-id {2} \
            --client-secret {3} \
            --location {4} \
            --api-model _output/azure-pytorch-elastic/apimodel.json \
            --new-node-count {5}\
            --apiserver azure-pytorch-elastic.{4}.cloudapp.azure.com"
    .format(args.subscription_id,
                    args.rg,
                    args.client_id,
                    args.client_secret,
                    args.location,
                    args.new_node_count)]
    run_commands(command)