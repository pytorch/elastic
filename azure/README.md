# Pytorch Elastic on Azure
This directory contains scripts and libraries that help users run pytorch elastic jobs on Azure.

## Prerequisites
1. Familiarity with [Azure](https://azure.microsoft.com/en-us/), [aks-engine](https://github.com/Azure/aks-engine), [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/)
2. Quota available for Standard_DS1_v2 instance and Standard_NC6s_v3 instance.
3. Access to Azure subscription, Resource Group and Storage. (Refer [here](https://github.com/Azure/MachineLearningNotebooks/blob/master/configuration.ipynb) for setup instructions).
4. Run the Azure login command `az login`.

# Sample Usage

1. #### Configure your job yaml and kubernetes.json
```
python petctl.py configure --name "test_job" --min_size 1 --max_size 5
```
This will create a spec for aks-engine instances and training job. Aks-engine launch [spec](config/kubernetes.json) is a simple json file that specifies the count and type of Rendezvous and Worker instances. A training job [spec](config/sample_specs.yaml) file is created with the specified job name and min, max worker count.
By default master node creates Standard DS1_v2 instance and worker nodes create Standard_NC6_v3 instances. Other Azure instances could be specified using --master_vm and --worker_vm options.

2. #### Setup your Kubernetes cluster

This step requires service prinicpal to create aks cluster.  
Instructions for generating service principal can be found at [portal](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal), [CLI](https://docs.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli?view=azure-cli-latest), [Powershell](https://docs.microsoft.com/en-us/powershell/azure/create-azure-service-principal-azureps).  
```
python petctl.py setup --dns_prefix azure-pytorch-elastic 
                       --rg "<resource_group>" 
                       --location "<location>" 
                       --subscription_id <subscription_id>
                       --client_id <service principal client id>
                       --client_secret <service principal client secret>                       
```
This creates an Azure Kubernetes cluster with 1 Standard_DS1_v2 master instances and specified number of Standard_NC6s_v3 worker instance in the resource group created in [Prerequisites](#Prerequisites) #3.

3. #### Upload to Azure Blob storage

This is an optional step to upload code and data to Azure blob storage. This step can be skipped if the training script and data are already available in Azure Blob storage.
```
python petctl.py upload_storage --source_path <path to files>
                                --account_name <storage account name>
                                --container_name <name of blob container>
                                --sas_token <SAS token for blob storage>
```
Instructions to generate SAS token are available [here](https://adamtheautomator.com/azure-sas-token/).

4. #### Generate Storage and Docker Image secrets

This step requires user blob storage account and docker image details.
Instructions for accessing storage account keys can be found at [portal](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage), [CLI](https://docs.microsoft.com/en-us/cli/azure/storage/account/keys)  

##### Generate Storage secret
```
python petctl.py storage_secret --account_name <storage account name> 
                                --account_key "<storage account key>" 
```
##### Generate Docker image secret
```
python petctl.py docker_secret --server <docker server> 
                               --username <docker username> 
                               --password <docker password>
                               --image_name <docker image name>
```

Training job file is updated to mount users storage account onto worker instances and apply the user provided docker image.
Base docker image to run Pytorch Elastic on Azure is at [Dockerfile](config/Dockerfile). Instructions on publishing docker image to  AzureContainer registry can be found at [ACR](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli).
Docker image secret generation can be skipped for running [imagenet](../../examples/imagenet/main.py) example as job specs yaml is already populated with public AzureML image with pytorch elastic support.

5. #### Start your training job

Submit the training job.
```
python petctl.py run_job
```
To run the provided imagenet example, training script and data can be uploaded to Azure blob storage by running
```
python petctl.py --upload_storage --source_path ../examples/imagenet/main.py
                                  --account_name <storage account name>
                                  --container_name code
                                  --sas_token <sas token for blob storage>
```
```
python petctl.py --upload_storage --source_path <path to imagenet train folder>
                                  --account_name <storage account name>
                                  --container_name data
                                  --sas_token <sas token for blob storage>
```

6. #### Check status of your job
```
python petctl.py check_status
```
7. #### Scale worker instances
```
python petctl.py scale --rg "<resource_group>"
                       --location "<location>"
                       --subscription_id <subscription_id>
                       --client_id <service principal client id>
                       --client_secret <service principal client secret>
                       --new_node_count <worker instances count>    
```
(Here subscription id and resource group is the one setup in [Prerequisites](#Prerequisites) #3.

8. #### Delete resources
````
python petctl.py delete_resources
````
This deletes the aks-engine cluster and all associated namespaces and secrets.
