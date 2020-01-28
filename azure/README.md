# Sample Usage

1. #### Setup your Kubernetes cluster
```
python petctl.py setup --dns_prefix aagarg-pytorch-elastic 
                       --rg "aagarg-rg" --location "WestEurope" 
                       --subscription_id 4aaa645c-5ae2-4ae9-a17a-84b9023bc56a 
                       --client_id 661a6fc2-ab3b-4cda-aaf6-78256cdba717 
                       --client_secret E]ltb:ATSZyIFAKI@GVv80m-Qh6RhnG6 
                       --storage_account_name petblob 
                       --storage_account_key "ejgV1NBvup6VIOK8jX00ZDeee/jzbxXGcgy4+L5BtcnAWQhMwHVA/be6qZupqJmP0qX2zKP8U9hq6PRofh26mA==" 
                       --docker_server petcr.azurecr.io 
                       --docker_username petCR 
                       --docker_password IgUIMK5MvK2rUHpDR2/v3hJ38z5ThmOG
```

2. #### Configure your job yaml and kubernetes.json
```
python petctl.py configure --name "test_job" --min_size 1--max_size 5
```

3. #### Start your training job
```
python petctl.py run_job
```
