## Trouble shooting

This doc is aimed at making controller work right.
It covers sections to debug why controller is not working in Kubernetes and the ways to debug your application failure.  


### Listing your controller pods 

The first thing to debug in your cluster is if your controller has been installed correctly.
Verify that all pods you expect to see are present and they're in `Ready` state. 


```shell
$ kubectl get all -n elastic-job
NAME                                              READY   STATUS    RESTARTS   AGE
pod/elastic-job-k8s-controller-79c568dcc9-kdbq2   1/1     Running   0          13m

NAME                                         READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/elastic-job-k8s-controller   1/1     1            1           27m

NAME                                                    DESIRED   CURRENT   READY   AGE
replicaset.apps/elastic-job-k8s-controller-79c568dcc9   1         1         1       13m
```

If something is wrong on the pod, you can run follow command to get detailed information.

```
 kubectl describe pod/elastic-job-k8s-controller-79c568dcc9-kdbq2 -n elastic-job
```

### Listing your cluster roles

TorchElastic Controller for Kubernetes needs to watch `Pods`, `Services`, `ElasticJob` etc to take action on events from resource change.
If you didn't apply cluster roles, role bindings or carelessly delete roles operator needs, you may see error like this. 


__cluster role is missing__
```shell
E0321 03:26:46.940863       1 reflector.go:125] Failed to list *v1alpha1.ElasticJob: elasticjobs.elastic.pytorch.org is forbidden: User "system:serviceaccount:elastic-job:default" cannot list resource "elasticjobs" in API group "elastic.pytorch.org" at the cluster scope: RBAC: clusterrole.rbac.authorization.k8s.io "elastic-job-k8s-controller-role" not found
E0321 03:26:46.940882       1 reflector.go:125] Failed to list *v1.Pod: pods is forbidden: User "system:serviceaccount:elastic-job:default" cannot list resource "pods" in API group "" at the cluster scope: RBAC: clusterrole.rbac.authorization.k8s.io "elastic-job-k8s-controller-role" not found
E0321 03:26:46.941037       1 reflector.go:125] Failed to list *v1.Service: services is forbidden: User "system:serviceaccount:elastic-job:default" cannot list resource "services" in API group "" at the cluster scope: RBAC: clusterrole.rbac.authorization.k8s.io "elastic-job-k8s-controller-role" not found
E0321 03:26:47.942903       1 reflector.go:125] Failed to list *v1alpha1.ElasticJob: elasticjobs.elastic.pytorch.org is forbidden: User "system:serviceaccount:elastic-job:default" cannot list resource "elasticjobs" in API group "elastic.pytorch.org" at the cluster scope: RBAC: clusterrole.rbac.authorization.k8s.io "elastic-job-k8s-controller-role" not found
``` 

__cluster role lack of permission__
```shell
E0323 00:48:47.713088       1 reflector.go:125] Failed to list *v1.Service: services is forbidden: User "system:serviceaccount:elastic-job:default" cannot list resource "services" in API group "" at the cluster scope
```

If you meet this problem, check your `clusterrole`, `clusterrolebinding` and `service account` exist.

```shell
kubectl get clusterroles  elastic-job-k8s-controller-role -o yaml
kubectl get clusterrolebindings elastic-job-k8s-controller-rolebinding -o yaml
kubectl get serviceaccount default -n elastic-job
```

If everything is there, please compare if `clusterroles` and `clusterrolebindings` match ones in [role.yaml](./config/rbac/role.yaml) and [role_binding.yaml](./config/rbac/role_binding.yaml)

### Looking at logs

If you create or delete a job and don't see any changes, it's possible job get stuck inside the controller with some issues. 
In this case, you have to check controller logs for more details.

```shell
kubectl logs -f elastic-job-k8s-controller-79c568dcc9-kdbq2 -n elastic-job
```

### Debug your applications

To debug applications that are deployed into Kubernetes and not behaving correctly, the first step is to check your job status. 

This is an example that both workers of job `imagenet` are stopped. 

```shell
$ kubectl get pods -n elastic-job
NAME                                          READY   STATUS    RESTARTS   AGE
elastic-job-k8s-controller-79c568dcc9-kdbq2   1/1     Running   0          38m
imagenet-worker-0                             0/1     Error     0          10m
imagenet-worker-1                             0/1     Error     0          10m
```
Let's check logs to see any other details we can get.

```shell
$ kubectl logs -f imagenet-worker-0 -n elastic-job

[INFO] 2020-03-23 00:47:23,416 main: rdzv init method=etcd://etcd-service:2379/imagenet?min_workers=1&max_workers=5&last_call_timeout=5
[INFO] 2020-03-23 00:47:23,417 main: Loading data from: /data/tiny-imagenet-200/train
[INFO] 2020-03-23 00:47:23,762 main: Loading model: resnet101
[INFO] 2020-03-23 00:47:29,745 main: Rank [0] running on GPU [0]
[WARNING] 2020-03-23 00:47:29,749 connectionpool: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f88a6d92c88>: Failed to establish a new connection: [Errno -2] Name or service not known',)': /v2/machines
[WARNING] 2020-03-23 00:47:29,750 connectionpool: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f88a6d92da0>: Failed to establish a new connection: [Errno -2] Name or service not known',)': /v2/machines
[WARNING] 2020-03-23 00:47:29,752 connectionpool: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f88a6d92e80>: Failed to establish a new connection: [Errno -2] Name or service not known',)': /v2/machines
[ERROR] 2020-03-23 00:47:29,754 client: Failed to get list of machines from http://etcd-service:2379/v2: MaxRetryError("HTTPConnectionPool(host='etcd-service', port=2379): Max retries exceeded with url: /v2/machines (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f88a6d92f98>: Failed to establish a new connection: [Errno -2] Name or service not known',))",)
Traceback (most recent call last):
  File "/opt/conda/lib/python3.6/site-packages/urllib3/connection.py", line 159, in _new_conn
    (self._dns_host, self.port), self.timeout, **extra_kw)
  File "/opt/conda/lib/python3.6/site-packages/urllib3/util/connection.py", line 57, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
  File "/opt/conda/lib/python3.6/socket.py", line 745, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno -2] Name or service not known
......
```

From the error message, we know that we use `etcd://etcd-service:2379` in the example as rdzv endpoint but `etcd-service:2379` is not reachable. 
Job failed because it can not connect to etcd. You need to double check liveness of etcd and try again.


### Bugs and Feature requests

If you have what looks like a bug, or you would like to make a feature request, please use the [GitHub issue tracking system](https://github.com/pytorch/elastic/issues).

Before you file an issue, please search existing issues to see if your issue is already covered.

If filing a bug, please include detailed information about how to reproduce the problem
