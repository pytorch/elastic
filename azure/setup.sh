#!/bin/sh

az login

# Install AKS-engine
curl -o get-akse.sh https://raw.githubusercontent.com/Azure/aks-engine/master/scripts/get-akse.sh
chmod 700 get-akse.sh
./get-akse.sh

# Deploy AKS cluster
aks-engine deploy --subscription-id $1 \
    --dns-prefix $2 \
    --resource-group $3 \
    --location $4 \
    --api-model config/kubernetes.json \
    --client-id $5 \
    --client-secret $6 \
    --set servicePrincipalProfile.clientId=$5 \
    --set servicePrincipalProfile.secret=$6


export KUBECONFIG=_output/aagarg-pytorch-elastic/kubeconfig/kubeconfig.westeurope.json

# Install Nvidia drivers
kubectl create namespace gpu-resources
kubectl apply -f nvidia-device-plugin-ds.yaml

# Create storage secrets
kubectl create secret generic pet-blob-secret --from-literal accountname=$7 --from-literal accountkey=$8 --type="azure/blobfuse"

# Install Blobfuse drivers
kubectl apply -f https://raw.githubusercontent.com/Azure/kubernetes-volume-drivers/master/flexvolume/blobfuse/deployment/blobfuse-flexvol-installer-1.9.yaml

# Create Docker image secret
kubectl create secret docker-registry pet-docker-secret --docker-server=$9 --docker-username=$10 --docker-password=$11
