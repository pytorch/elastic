## Torch elastic k8s controller

## Overview

Torch elastic k8s controller managed a Kubernetes custom resource `ElasticJob` and makes it easy to
run Torch Elastic workloads on Kubernetes.   

### Prerequisites

- Kubernetes >= 1.12
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl)
- [kustomize](https://github.com/kubernetes-sigs/kustomize/blob/master/docs/INSTALL.md)

### Use Amazon EKS to create a Kubernetes cluster

We highly recommend to use eksctl to create Amazon EKS cluster. This process will take 10~15 minutes. 
Use other instance type if you don't want to use GPU.   

```shell
eksctl create cluster \
    --name=torchelastic \
    --node-type=p3.2xlarge \
    --region=us-west-2 \
    --version=1.15 \
    --ssh-access \
    --ssh-public-key=~/.ssh/id_rsa.pub \
    --nodes=2
```

> Note: EKS is not required to run this controller, you can use other Kubernetes clusters. 
> This controller has been tested running well on EKS.

### Install Nvidia Device Plugin
In order to enable GPU support in your EKS cluster, deploy the following Daemonset:

```shell
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta4/nvidia-device-plugin.yml
```

### Installing the ElasticJob CRD and operator on your k8s cluster

```shell
kustomize build config/default  | kubectl apply -f -
```

or 
```shell
kubectl apply -k config/default
```

You will see logs like following

```yaml
$ kustomize build config/default | kubectl apply -f -

namespace/elastic-job created
customresourcedefinition.apiextensions.k8s.io/elasticjobs.elastic.pytorch.org created
role.rbac.authorization.k8s.io/leader-election-role created
clusterrole.rbac.authorization.k8s.io/manager-role created
rolebinding.rbac.authorization.k8s.io/leader-election-rolebinding created
clusterrolebinding.rbac.authorization.k8s.io/elastic-job-k8s-controller-rolebinding created
deployment.apps/elastic-job-k8s-controller created
```

Verify that the ElasticJob custom resource is installed

```shell
kubectl get crd
```

The output should include `elasticjobs.elastic.pytorch.org`

```
NAME                                              CREATED AT
...
elasticjobs.elastic.pytorch.org                   2020-03-18T07:40:53Z
...
```

Verify controller is ready 

```shell
kubectl get pods -n elastic-job

NAME                                          READY   STATUS    RESTARTS   AGE
elastic-job-k8s-controller-6d4884c75b-z22cm   1/1     Running   0          15s
```

### Check logs of controller

```yaml
kubectl logs -f elastic-job-k8s-controller-6d4884c75b-z22cm -n elastic-job

2020-03-19T10:13:43.532Z	INFO	controller-runtime.metrics	metrics server is starting to listen	{"addr": ":8080"}
2020-03-19T10:13:43.534Z	INFO	controller-runtime.controller	Starting EventSource	{"controller": "elasticjob", "source": "kind source: /, Kind="}
2020-03-19T10:13:43.534Z	INFO	controller-runtime.controller	Starting EventSource	{"controller": "elasticjob", "source": "kind source: /, Kind="}
2020-03-19T10:13:43.534Z	INFO	controller-runtime.controller	Starting EventSource	{"controller": "elasticjob", "source": "kind source: /, Kind="}
2020-03-19T10:13:43.534Z	INFO	setup	starting manager
2020-03-19T10:13:43.534Z	INFO	controller-runtime.manager	starting metrics server	{"path": "/metrics"}
2020-03-19T10:13:43.822Z	DEBUG	controller-runtime.manager.events	Normal	{"object": {"kind":"ConfigMap","namespace":"elastic-job","name":"controller-leader-election-helper","uid":"50269b8b-69ca-11ea-b995-0653198c16be","apiVersion":"v1","resourceVersion":"2107564"}, "reason": "LeaderElection", "message": "elastic-job-k8s-controller-6d4884c75b-z22cm_4cf549b7-3289-4285-8e64-647d067178bf became leader"}
2020-03-19T10:13:44.021Z	INFO	controller-runtime.controller	Starting Controller	{"controller": "elasticjob"}
2020-03-19T10:13:44.121Z	INFO	controller-runtime.controller	Starting workers	{"controller": "elasticjob", "worker count": 1}
```

### Deploy a ElasticJob

In the following example, training container will download training code from S3. Your need to attach `AmazonS3ReadOnlyAccess` 
policy to your node group role to grant the permission.
 
Create an etcd instance and service for rdzvEndpoint, it will expose a Kubernetes service `etcd-service` with port `2379`.
```
kubectl apply -f config/samples/etcd.yaml
``` 
 
Build your own trainer image 

```
export DOCKERHUB_USER=<your_dockerhub_username>
cd kubernetes/config/samples

docker build -t $DOCKERHUB_USER/examples:imagenet .
docker push $DOCKERHUB_USER/examples:imagenet
```

Update `config/samples/imagenet.yaml` to use your image, scripts and rdzvEndpoint. 


Submit the training job. 
```yaml
kubectl apply -f config/samples/imagenet.yaml
```

As you can see, training pod and headless services have been created.
```yaml
$ kubectl get pods -n elastic-job
NAME                                          READY   STATUS    RESTARTS   AGE
elastic-job-k8s-controller-6d4884c75b-z22cm   1/1     Running   0          11m
imagenet-worker-0                             1/1     Running   0          5s
imagenet-worker-1                             1/1     Running   0          5s

$ kubectl get svc -n elastic-job
NAME                TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)     AGE
imagenet-worker-0   ClusterIP   None         <none>        10291/TCP   34s
imagenet-worker-1   ClusterIP   None         <none>        10291/TCP   34s
```

You can adjust desired replica `.spec.replicaSpecs[Worker].replicas` and apply change to k8s. 
```
kubectl apply -f config/samples/imagenet.yaml
```

### Monitoring jobs

You can describe the job to check job status and job related events.
In following example, `imagenet` job is created in `elastic-job` namespace, change to use your job name and namespace in your command.

```
kubectl describe elasticjob imagenet -n elastic-job

Name:         imagenet
Namespace:    elastic-job
Labels:       <none>
Annotations:  kubectl.kubernetes.io/last-applied-configuration:
                {"apiVersion":"elastic.pytorch.org/v1alpha1","kind":"ElasticJob","metadata":{"annotations":{},"name":"imagenet","namespace":"elastic-job"}...
API Version:  elastic.pytorch.org/v1alpha1
Kind:         ElasticJob
Metadata:
  Creation Timestamp:  2020-03-19T10:30:55Z
  Generation:          5
  Resource Version:    2110451
  Self Link:           /apis/elastic.pytorch.org/v1alpha1/namespaces/elastic-job/elasticjobs/imagenet
  UID:                 b6f6b7ae-69cc-11ea-b995-0653198c16be
Spec:
  Run Policy:
  Max Replicas:   5
  Min Replicas:   1
  Rdzv Endpoint:  etcd-service:2379
  Replica Specs:
    Worker:
      Replicas:        2
      Restart Policy:  ExitCode
      Template:
        Metadata:
          Creation Timestamp:  <nil>
        Spec:
          Containers:
            Args:
              s3://torchelastic-shjiaxin-1h71m-s3bucket-m1b9b9pjldqw/petctl/shjiaxin/imagenet-job/main.py
              --input_path
              /data/tiny-imagenet-200/train
              --epochs
              10
            Image:              seedjeffwan/examples:0.1.0rc1
            Image Pull Policy:  Always
            Name:               elasticjob-worker
            Ports:
              Container Port:  10291
              Name:            elasticjob-port
            Resources:
              Limits:
                nvidia.com/gpu:  1
Status:
  Conditions:
    Last Transition Time:  2020-03-19T10:30:55Z
    Last Update Time:      2020-03-19T10:30:55Z
    Message:               ElasticJob imagenet is running.
    Reason:                ElasticJobRunning
    Status:                True
    Type:                  Running
  Replica Statuses:
    Worker:
      Active:  3
Events:
  Type    Reason                   Age   From                    Message
  ----    ------                   ----  ----                    -------
  Normal  SuccessfulCreatePod      13s   elastic-job-controller  Created pod: imagenet-worker-0
  Normal  SuccessfulCreatePod      13s   elastic-job-controller  Created pod: imagenet-worker-1
  Normal  SuccessfulCreatePod      13s   elastic-job-controller  Created pod: imagenet-worker-2
  Normal  SuccessfulCreateService  13s   elastic-job-controller  Created service: imagenet-worker-0
  Normal  SuccessfulCreateService  13s   elastic-job-controller  Created service: imagenet-worker-1
  Normal  SuccessfulCreateService  13s   elastic-job-controller  Created service: imagenet-worker-2

```

### Trouble Shooting

Please check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
