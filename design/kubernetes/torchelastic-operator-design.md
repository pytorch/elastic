# TorchElastic Operator on Kubernetes

## Background

PyTorch continues to be used for the latest state-of-the-art research, making up nearly 70% of [papers](https://chillee.github.io/pytorch-vs-tensorflow/) that cite a framework.

The current PyTorch Distributed Data Parallel (DDP) module enables data parallel training where each process trains the same model but on different shards of data. It enables bulk synchronous, multi-host, multi-GPU/CPU execution of ML training. However, DDP has several shortcomings; e.g.jobs cannot start without acquiring all the requested nodes; jobs cannot continue after a node fails due to an error or transient issue; jobs cannot incorporate a node that joined later; and lastly; progress cannot be made with the presence of a slow/stuck node.

The focus of [PyTorch Elastic](https://github.com/pytorch/elastic), which uses Elastic Distributed Data Parallelism, is to address these issues and build a generic framework/APIs for PyTorch to enable reliable and elastic execution of these data parallel training workloads. It will provide better programmability, higher resilience to failures of all kinds, higher-efficiency and larger-scale training compared with pure DDP.

## Motivation

With job fault tolerance and elastic training, we can unlock a lot of features.

Users can enable job priority and preemption in the cluster. Losing a task becomes acceptable and the user won't lose the entire job progress. More importantly, it will help guarantee SLAs of crititcal jobs, even in a cluster under resource pressure.

Cost and GPU utilization will be further optimized with this feature, since users can launch jobs with partial resources and spot GPU instances can be used as well without worrying.

## User Experience

* Users should define the `minReplicas` and `maxReplicas` number of tasks of a job instead of a fixed number. The TorchElastic Operator will launch jobs in Kubernetes, setup the needed network topology and manage the job lifecycle.
* Users need to specify the etcd endpoint used as the RDZV service for task coordination.
* The desired `spec.replicaSpecs[Worker].replicas`, being number of tasks, has to be within the range from `minReplicas` to `maxReplicas`.
* Users can easily create/delete a torch elastic job using `kubectl` using a job manifest.
* Users are able to describe custom resources to monitor the job status.

## High Level Design

Workers in torch elastic job are equivalent and their communication is peer to peer. In this case, every pod should be able to talk with every other pod, and we need to create a `headless` service for every pod. Once the job is done, controller won't terminate any pods, user can check logs for any worker. Manual job deletion will delete all pods belong to it.

A config with kind `ElasticJob` defines the job spec and the controller will reconcile against this definition. It will create/update/delete pods and services if there are any changes in the job orin the kubernetes resources (pods, services) changes owned by `ElasticJob`.

```
apiVersion: "elastic.pytorch.org/v1alpha1"
kind: "ElasticJob"
metadata:
  name: "classy-vision-job"
spec:
  rdzvEndpoint: "etcd-service:2379"
  minReplicas: 2
  maxReplicas: 5
  replicaSpecs:
    Worker:
      replicas: 3
      restartPolicy: ExitCode
      template:
        apiVersion: v1
        kind: Pod
        spec:
          containers:
          - name: torchelasticworker
            image: torchelastic/examples:0.1.0rc1
            imagePullPolicy: Always
            args:
              - "s3://code_path/petctl/user/my_job/main.py"
              - --config_file
              - "/data/classy_vision/resnet50_synthetic_image_classy_config.json"
              - "--checkpoint_folder"
              - "/data/classy_vision/checkpoint""

```

*Network Communication*

In this case, every pod should be able to talk with each other and we need to create headless service for every pod since they use hostname registered in rdzv endpoint to find peers.

*Failure condition*

Torch Elastic controller will only fail a job if active workers is under minReplicas size user specified. Otherwise, it will try to reschedule failed pods and maintain the desired task size.

*rdzvEndpoint*

`rdzvEndpoint` needs to be specified by user. It could be high available etcd quorum or single etcd pod on Kubernetes cluster.

*Replicas*

`replicas` represents the desired task size. Torch elastic job doesn't need all the workers to be ready to start training. We can set this field to job.spec.maxReplicas and try to allocate more resources. If cluster doesn't have enough resources, some tasks maybe pending and job can still start.


These are the resources the controller creates from a `TorchElasticJob`:

**Pod**

```
apiVersion: v1
kind: Pod
metadata:
  name: classy-vision-job-worker-${index}
  labels:
    job-name: classy-vision-job
    group-name=elastic.pytorch.org
    replica-index: 0
    replica-type=worker
spec:
  containers:
    image: torchelastic/examples:0.1.0rc1
    imagePullPolicy: Always
    name: torchelasticworker
    env:
      - name: RDZV_ENDPOINT
        value: "etcd-:2379"
      - name: JOB_ID
        value: "classy-vision-job"
      - name: SIZE
        value: "3"
      - name: MIN_SIZE
        value: "2"
      - name: MAX_SIZE
        value: "5"
  restartPolicy: OnFailure
```

**Service**

```
apiVersion: v1
kind: Service
metadata:
  name: classy-vision-job-worker-${index}
spec:
  selector:
    job-name: classy-vision-job
    group-name=elastic.pytorch.org
    replica-index: 0
    replica-type=worker
  clusterIP: None
```

**Job Status**

``` yaml
kubectl describe elasticjob classy-vision-job
Name:         classy-vision-job
Namespace:    default
API Version:  elastic.pytorch.org/v1alpha1
Kind:         ElasticJob
Spec:
  ...
Status:
  Conditions:
    Last Transition Time:  2020-01-22T23:10:44Z
    Last Update Time:      2020-01-22T23:10:44Z
    Message:               job classy-vision-job is created.
    Reason:                ElasticJobCreated
    Status:                True
    Type:                  Created
    Last Transition Time:  2020-01-22T23:10:49Z
    Last Update Time:      2020-01-22T23:10:49Z
    Message:               ElasticJob classy-vision-job is running.
    Reason:                ElasticJobRunning
    Status:                False
    Type:                  Running
    Last Transition Time:  2020-01-22T23:10:49Z
    Last Update Time:      2020-01-22T23:10:49Z
    Message:               ElasticJob classy-vision-job is failed because 2 workers replica(s) failed.
    Reason:                ElasticJobFailed
    Status:                True
    Type:                  Failed
  Replica Statuses:
    Worker:
      Active:  1
      Failed:  2
Events:
  Type     Reason                     Age                From                     Message
  ----     ------                     ----               ----                     -------
  Normal   SuccessfulCreatePod        39m                elastic-job-controller   Created pod: classy-vision-job-worker-0
  Normal   SuccessfulCreatePod        39m                elastic-job-controller   Created pod: classy-vision-job-worker-1
  Normal   SuccessfulCreateService    39m                elastic-job-controller   Created service: classy-vision-job-worker-0
  Normal   SuccessfulCreateService    39m                elastic-job-controller   Created service: classy-vision-job-worker-1
  Normal   ExitedWithCode             39m (x3 over 39m)  elastic-job-controller   Pod: default.classy-vision-job-worker-0 exited with code 1
  Warning  ElasticJobRestarting       39m (x3 over 39m)  elastic-job-controller   ElasticJob classy-vision-job is restarting because 1 Worker replica(s) failed.
  Normal   ElasticJobFailed           39m                elastic-job-controller   ElasticJob classy-vision-job is failed because 2 Worker replica(s) failed.
```

## Not in scope

TorchElastic Operator can simplify the setups to run torch elastic jobs and manage entire job lifecycle. It is hard for the controller to monitor cluster resources and dynamically adjust the task size. Instead, having a separate component like a batch scheduler to make the decision is a better option at this stage, to limit the scope of this project.

Currently, operator has to accept an etcd service as `rdzvEndpoint`. We may consider to make this field optional and provide etcd service by controller if it's not set.