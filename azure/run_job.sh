#!/bin/sh

# Reset if application is already running
kubectl delete -f config/azure-pytorch-elastic.yaml

# Run application
kubectl apply -f config/azure-pytorch-elastic.yaml

# Monitor application
kubectl describe pods
kubectl get pods --selector app=azure-pytorch-elastic
