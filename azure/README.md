# Sample Usage

1. #### Configure your job yaml and kubernetes.json
```
python petctl.py configure --name "test_job" --min_size 1 --max_size 5
```

2. #### Setup your Kubernetes cluster
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

3. #### Start your training job
```
python petctl.py run_job
```
