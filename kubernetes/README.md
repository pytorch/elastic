## TorchElastic Controller for Kubernetes

## Overview

TorchElastic Controller for Kubernetes manages a Kubernetes custom resource `ElasticJob` and makes it easy to
run Torch Elastic workloads on Kubernetes.   

### Prerequisites

- Kubernetes >= 1.12
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl)
- [kustomize](https://github.com/kubernetes-sigs/kustomize/blob/master/docs/INSTALL.md)

> **NOTE**: 
>
>  1. (recommended) create a cluster with GPU instances as some examples
>     (e.g. imagenet) only work on GPU.
>  2. If you provision instances with a single GPU you will only be able to run
>     a single worker per node.
>  3. Our examples assume 1 GPU per node so you will have to adjust 
>     `--nproc_per_node` to be equal to the number of CUDA devices
>     on the instance you are using if you want to run multiple workers per 
>     container

### (Optional) Setup

Here we provide the instructions to create an Amazon EKS cluster. If you 
are not using AWS please refer to your cloud/infrastructure provider's manual
to setup a kubernetes cluster. 

> **NOTE**: EKS is not required to run this controller, 
>  you can use other Kubernetes clusters.

Use `eksctl` to create an Amazon EKS cluster. This process takes ~15 minutes. 

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

Install Nvidia device plugin to enable GPU support on your cluster.
Deploy the following Daemonset:

```shell
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta4/nvidia-device-plugin.yml
```

### Install `ElasticJob` controller and CRD 


```shell

git clone https://github.com/pytorch/elastic.git
cd elastic/kubernetes

kubectl apply -k config/default 
# or
# kustomize build config/default  | kubectl apply -f -
```

You will see logs like following

```shell
namespace/elastic-job created
customresourcedefinition.apiextensions.k8s.io/elasticjobs.elastic.pytorch.org created
role.rbac.authorization.k8s.io/leader-election-role created
clusterrole.rbac.authorization.k8s.io/manager-role created
rolebinding.rbac.authorization.k8s.io/leader-election-rolebinding created
clusterrolebinding.rbac.authorization.k8s.io/elastic-job-k8s-controller-rolebinding created
deployment.apps/elastic-job-k8s-controller created
```

Verify that the `ElasticJob` custom resource is installed

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

1. Deploy an etcd server. This will expose a Kubernetes service `etcd-service` with port `2379`.
    ```
    kubectl apply -f config/samples/etcd.yaml
    ```
1. Get the etcd server endpoint
   ```
   $ kubectl get svc -n elastic-job

   NAME           TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
   etcd-service   ClusterIP   10.100.104.168   <none>        2379/TCP   5m5s
   ```
   
1. Update `config/samples/<imagenet.yaml|classy_vision.yaml>`:
    1. set `rdzvEndpoint` (e.g. `10.100.104.168:2379`) to the etcd server you just provisioned.
    1. set `minReplicas` and `maxReplicas` to the desired min and max num nodes
       (max should not exceed your cluster capacity)
    1. set `Worker.replicas` to the number of nodes to start with (you may 
       modify this later to scale the job in/out)
    1. set the correct `--nproc_per_node` in `container.args` based on the
       instance you are running on.
     
    > **IMPORTANT** A `Worker` in the context of kubernetes refers to `Node` in
      `torchelastic.distributed.launch`. Each kubernetes `Worker` can run multiple
       trainers processes (a.k.a `worker` in `torchelastic.distributed.launch`).

1. Submit the training job.

    ```
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

1. You can scale the number of nodes by adjusting 
   `.spec.replicaSpecs[Worker].replicas` and applying the change.
    ```
    kubectl apply -f config/samples/imagenet.yaml
    ```
    
    > **NOTE** since you are scaling the containers, you will be scaling in 
      increments of `nproc_per_node` trainers. In our case ``--nproc_per_node=1``
      For better performance consider using an instance with multiple 
      GPUs and setting `--nproc_per_node=$NUM_CUDA_DEVICES`.
   
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
              /workspace/examples/imagenet/main.py
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
