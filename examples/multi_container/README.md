# A minimal elastic agent example
In this example, we show how to use the PyTorch Elastic Trainer launcher to start a distributed application in an elastic and fault tolerant manner. The application is intentionally kept "bare bones" since the objective is to show how to create a `torch.distributed.ProcessGroup` instance. Once a `ProcessGroup` is created, you can use any functionality needed from the `torch.distributed` package.

This application can be run on practically any machine that supports Docker containers and does not require installing additional software or modifying your existing Python environment.

> The `docker-compose.yml` file is based on the example provided with the [Bitnami ETCD container image](https://hub.docker.com/r/bitnami/etcd/).

## Prerequisites
We assume you have a recent version of Docker (version 18.03 or above) and Docker Compose installed on your machine. Verify the version by running
```
docker --version
```
and
```
docker-compose --version
```
which should print something like
```
Docker version 19.03.8, build afacb8b
```
and
```
docker-compose version 1.25.4, build 8d51620a
```
respectively.
## Obtaining the example repo
Clone the PyTorch Elastic Trainer Git repo using
```
git clone https://github.com/pytorch/elastic.git
```
make an environment variable that points to the elastic repo, e.g.
```
export TORCHELASTIC_HOME=~/elastic
```

# Building the samples Docker container
While you can run the rest of this example using a pre-built Docker image, you can also build one for yourself. This is especially useful if you would like to customize the image. To build the image, run:
```
cd $TORCHELASTIC_HOME && docker build -t hello_elastic:dev .
```

# Running an existing sample
This example uses `docker-compose` to run two containers: one for the ETCD service and one for the sample application itself. Docker compose takes care of all aspects of establishing the network interfaces so the application container can communicate with the ETCD container.

To start the example, run
```
cd $TORCHELASTIC_HOME/examples/multi_container && docker-compose up
```
You should see two sets of outputs, one from ETCD starting up and one from the application itself. The output from the application looks something like this:

```
example_1      | INFO 2020-04-03 17:36:31,582 Etcd machines: ['http://etcd-server:2379']
example_1      | *****************************************
example_1      | Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
example_1      | *****************************************
example_1      | INFO 2020-04-03 17:36:31,922 Attempting to join next rendezvous
example_1      | INFO 2020-04-03 17:36:31,929 New rendezvous state created: {'status': 'joinable', 'version': '1', 'participants': []}
example_1      | INFO 2020-04-03 17:36:32,032 Joined rendezvous version 1 as rank 0. Full state: {'status': 'frozen', 'version': '1', 'participants': [0], 'keep_alives': []}
example_1      | INFO 2020-04-03 17:36:32,032 Waiting for remaining peers.
example_1      | INFO 2020-04-03 17:36:32,033 All peers arrived. Confirming membership.
example_1      | INFO 2020-04-03 17:36:32,116 Waiting for confirmations from all peers.
example_1      | INFO 2020-04-03 17:36:32,118 Rendezvous version 1 is complete. Final state: {'status': 'final', 'version': '1', 'participants': [0], 'keep_alives': ['/torchelastic/p2p/run_None/rdzv/v_1/rank_0'], 'num_workers_waiting': 0}
example_1      | INFO 2020-04-03 17:36:32,118 Creating EtcdStore as the c10d::Store implementation
example_1      | ======================================================
example_1      | Environment variables set by the agent on PID 51:
example_1      | {'GROUP_RANK': '0',
example_1      |  'LOCAL_RANK': '1',
example_1      |  'MASTER_ADDR': '6002aeb7c496',
example_1      |  'MASTER_PORT': '46289',
example_1      |  'RANK': '1',
example_1      |  'TORCHELASTIC_MAX_RESTARTS': '100',
example_1      |  'TORCHELASTIC_RESTART_COUNT': '0',
example_1      |  'WORLD_SIZE': '2'}
example_1      | ======================================================
example_1      |
example_1      | ======================================================
example_1      | Environment variables set by the agent on PID 52:
example_1      | {'GROUP_RANK': '0',
example_1      |  'LOCAL_RANK': '0',
example_1      |  'MASTER_ADDR': '6002aeb7c496',
example_1      |  'MASTER_PORT': '46289',
example_1      |  'RANK': '0',
example_1      |  'TORCHELASTIC_MAX_RESTARTS': '100',
example_1      |  'TORCHELASTIC_RESTART_COUNT': '0',
example_1      |  'WORLD_SIZE': '2'}
example_1      | ======================================================
example_1      |
example_1      | On PID 51, after init process group, rank=1, world_size = 2
example_1      |
example_1      | On PID 52, after init process group, rank=0, world_size = 2
example_1      |
lib_example_1 exited with code 0
```
As you can see above, the application starts a process group with two peers and all information needed to initialize the group is provided by the elastic agent via environment variables.

## Customizing parameters
The application Docker container can run an arbitrary user-provided script and run . To do this, simply mount the path containing your script into the application container by creating a volume mount in the `docker-compose.yaml` file as shown below:

```yaml
  example:
    image: 'hello_elastic:dev'
    volumes:
       - /path/to/your/app:/workspace
    command: --nnode=1 --nproc_per_node=2 --rdzv_endpoint=etcd-server /workspace/your_app.py
    networks:
      - app-tier
```
Note that the path to your application (`/path/to/your/app` in the example above) must be an absolute path to the directory containing your script (`your_app.py` above)

## Conclusions
In this simple example, we illustrated the following principles when using PyTorch Elastic Trainer:
1. How to launch a PyTorch distributed application using the elastic launcher.
2. How to obtain parameters such as the world size, local rank and the master URL within an application to establish the process group.
3. How to configure parameters for an elastic job such as the number of workers per node and the number of times your application should be restarted in the event of failures.

In the next set of samples, we will cover more advanced topics such as checkpointing state in your application and deploying it to an orchestrator such as Kubernetes.
