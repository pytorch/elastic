# Pytorch Elastic on Azure
This directory contains scripts and libraries that help users run pytorch elastic jobs on Azure.

## Prerequisites
1. Familiarity with [Azure](https://azure.microsoft.com/en-us/), [aks-engine](https://github.com/Azure/aks-engine), [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/)
2. Install the Azure [CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
3. Quota available for Standard_DS1_v2 instance and Standard_NC6s_v3 instance.

# Sample Usage

1. #### Configure your job yaml and kubernetes.json
```
python petctl.py configure --name "test_job" --min_size 1 --max_size 5
```
This will create a spec for aks-engine instances and training job. Aks-engine launch [spec](config/kubernetes.json) is a simple json file that specifies the count and type of Rendezvous and Worker instances. A training job [spec](config/sample_specs.yaml) file is created with the specified job name and min, max worker count.

2. #### Setup your Kubernetes cluster

This step requires service prinicpal and blob storage account details.  
Instructions for generating service principal can be found at [portal](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal), [CLI](https://docs.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli?view=azure-cli-latest), [Powershell](https://docs.microsoft.com/en-us/powershell/azure/create-azure-service-principal-azureps).  
Instructions for accessing storage account keys can be found at [portal](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage), [CLI](https://docs.microsoft.com/en-us/cli/azure/storage/account/keys)  

```
python petctl.py setup --dns_prefix azure-pytorch-elastic 
                       --rg "<resource_group>" 
                       --location "<location>" 
                       --subscription_id <subscription_id>
                       --client_id <service principal client id>
                       --client_secret <service principal client secret>
                       --storage_account_name <storage account name> 
                       --storage_account_key "<storage account key>" 
                       --docker_server <docker server> 
                       --docker_username <docker username> 
                       --docker_password <docker password>
```
This creates an Azure Kubernetes cluster with 1 Standard_DS1_v2 master instances and specified number of Standard_NC6s_v3 worker instances. Training job file is updated to mount users storage account onto worker instances.

3. #### Start your training job
```
python petctl.py run_job
```

4. #### Check status of your job
```
python petctl.py check_status
```
5. #### Scale worker instances
```
python petctl.py scale --rg "<resource_group>"
                       --location "<location>"
                       --subscription_id <subscription_id>
                       --client_id <service principal client id>
                       --client_secret <service principal client secret>
                       --new_node_count <worker instances count>    
```
6. #### Delete resources
````
python petctl.py delete_resources
````
This deletes the aks-engine cluster and all associated namespaces and secrets.