# torchelastic on AWS
This directory contains scripts and libraries that help users run torchelastic
jobs on AWS.

## Prerequisites

1. Familiarity with basic AWS (EC2, Auto Scaling Groups, S3, EFS).
2. (suggested) install and setup [`awscli`](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).
3. Basic knowledge of containers (we use Docker in our examples).

## Requirements

1. `pip install boto3`
2. `git clone https://github.com/pytorch/elastic.git`
3. The following AWS resources:
    1. EC2 [instance profile](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2_instance-profiles.html)
    2. EC2 [key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
    3. [Subnet(s)](https://docs.aws.amazon.com/vpc/latest/userguide/default-vpc.html#create-default-subnet)
    4. [Security group](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html#DefaultSecurityGroup)
    5. EFS volume


## Quickstart

This guide shows how to get a simple torchelastic job running on AWS. `petctl`
is a commandline tool that helps run distributed jobs written with torchelastic
on EC2 instances.

### Create specs file
First, lets create a launch spec. This is a simple json file that specifies
the launch configuration of EC2 instances. We have included a
[sample specs file](config/sample_specs.json) so make a copy and fill it in.
You only need to fill in the fields with `<YOUR XXXXX>`, you can leave the other
fields alone for now.

```
# cd $torchelastic_repository_root
mkdir ~/torchelastic_workspace
cp aws/config/sample_specs.json ~/torchelastic_workspace/specs.json
```

The specs file is divided into two sections: `rdzv` and `worker`. As their names
imply the `rdzv` section contains the launch specs for the instances
of the rendezvous backend (e.g. etcd). The `worker` section contains the launch
specs for the worker instances.

The following subsections describe the fields in the specs file.

#### Instance Type and Accelerator
```
    "instance_type" : "[ec2 instance type]",
    "accelerator" : "[none | gpu]",
```
The instance type specifies the EC2 instance type to use. The `accelerator`
field can either be `none` or `gpu`. If an EC2 instance that has GPU capability
is specified (e.g. `g3`, `p2`, `p3` instance families) then you must specify
`accelerator = gpu`. 

> If `accelerator=gpu` is not specified on a GPU capable instance type,
`petctl` assumes you will only use CPU and will use an AMI that does not have
CUDA nor NVIDIA drivers.

#### Subnet
Note that you can add multiple subnets. Each subnet belongs to an availability zone (AZ)
so you can spread your instances across AZs by specifying multiple subnets:

```
    "subnets" : [
      "[subnet_in_us-west-2a]",
      "[subnet_in_us-west-2b]",
      ...
    ],
```
> Some instances are not available in all AZs, make sure to create the subnet
in the AZ that supports the instance type that you plan to run your jobs on.

#### Security Group
torchelastic jobs are distributed and hence require nodes to communicate with each
other. Make sure that your security group allows all inbound traffic between
instances within the same security group.
> Optionally you may want to allow inbound SSH access in case you need to log
into the instance for debugging.

#### EC2 Instance Profile
```
    "instance_role" : "[ec2 instance profile]",
```
This is the IAM role that is used when accessing other AWS services from **within**
the EC2 instance (e.g. accessing S3 from the worker host). To learn more about
instance profiles refer to the
AWS [documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2_instance-profiles.html).

For this example we will require S3 Read Only access from the workers since
we will use S3 to upload your local script and download it on the worker-side.

> You may wish to add other privileges to this role depending on what your workers
do. For instance, if your job writes output to S3, then you will have to attach
S3 Write IAM policy to this profile.

#### S3 Bucket and Prefix
`petctl` uploads your local script to S3 and pulls it down on the worker.
Specify the S3 bucket and prefix for this purpose:
```
    "s3_bucket" : "<YOUR S3 BUCKET NAME>",
    "s3_prefix" : "<YOUR S3 PREFIX>"
```

#### Additional Worker Specs
Workers have a couple of additional specs compared to rdzv.

##### Docker Image
```
"docker_image" : "torchelastic/aws:0.1.0-rc",
```

Note that the `worker` section in the specs file has an extra `docker_image`
configuration. This is because the workers run in a docker container whereas
the rendezvous backend (etcd) runs directly on the instance. The
`torchelastic/aws` image contains torchelastic (along with all of its dependencies)
and a few runtime utilities such as the `fetch_and_run` script that allows us
to run arbitrary user scripts. For production, you may consider creating your
own docker image with a custom `ENTRYPOINT` specific to your application.

##### EFS
An Elastic File System volume is mounted on each
worker instance (it is mounted **through** all the way to the container).
EFS acts much like NFS in terms of semantics. Use it as if you were using NFS.
You may store your input dataset, store model checkpoints, or job outputs here.

> The specified EFS volume is mounted on `/mnt/efs1`. On the host and container.

### Configure `petctl`
After creating a specs file, configure `petctl`

```
python3 petctl.py configure
```

This will prompt for the **absolute** path of your specs file and the AWS region.
You can run the `configure` sub-command as many times as you wish in case you made
a mistake or you need to reconfigure `petctl`. 

You'll notice that after configuration is done, there is a config file generated
under `$HOME/.petctl`.

> This is similar to how the aws cli is configured. If you decide to skip
configuration, then you **must** pass `--specs_file` and `--region` arguments
to `petctl` each time (e.g. `petctl --sepcs_file /home/$USER/specs.json --region us-west-2`).

### Write a script
If you already have a script that uses torchelastic to run distributed training,
great! Otherwise you can use the provided [imagenet example](../examples/imagenet/main.py)
or [classy vision](../examples/classy_vision/main.py). Or... this is a great time to work on one.

> If you are using `examples/imagenet/main.py` you must download the imagenet
dataset from [here](http://image-net.org/download) onto the EFS volume you specified.
The dataset will be available to the workers on `/mnt/efs1/<download_dir>`.

### Run the script

We will assume that you are working with the imagenet example.
To run the script we'll use `petctl`,

``` bash
python3 petctl.py run_job --size 2 --name ${USER}-job examples/imagenet/main.py -- model-arch resnet101
```

In the example above, the named arguments, such as, `--size` and `--specs_file` are 
self explanatory and are arguments supplied to `petctl`. The positional arguments have the form:

```
[local script] -- [script args ...]
  -- or -- 
[local directory] -- [script] [script args...]
```

If the first positional argument is a path to a script file, then the script
is uploaded to S3 and the script arguments specified after the `--` delimiter
are passed through to the script.

If the first positional argument is a directory, then a tarball of the directory
is created and uploaded to S3 and is extracted on the worker-side. In this case
the first argument after the `--` delimiter is the path to the script **relative** to the
specified directory and the rest of the arguments after the delimiter is passed 
to the script.

In our example we specified
```
petctl.py run_job [...] examples/imagenet/main.py -- --model_arch resenet 101
```

We could have decided to specify the directory instead
```
petctl.py run_job [...] examples/imagenet -- main.py --model_arch resenet 101
```

(TIP) Besides a local script or directory you can run with scripts or `tar` files
that have already been uploaded to S3 or directly point it to a file or directory
on the container.
``` bash
python3 petctl.py run_job [...] s3://my-bucket/my_script.py
python3 petctl.py run_job [...] s3://my-bucket/my_dir.tar.gz -- my_script.py

# or
python3 petctl.py run_job [...] --no_upload /path/in/container/dir -- my_script.py
python3 petctl.py run_job [...] --no_upload /path/in/container/dir/my_script.py
```

Once the `run_job` command returns log into the EC2 console, you will see two
Auto Scaling Groups
1. etcd server 
2. workers

The workers run in a docker container. You can take a look at their console outputs by running

``` bash
# get the container id
docker ps
# tail the container logs
docker logs -f <container id>
```
> Note: by design, `petctl` tries to use the least number of AWS services. This
was done intentionally to allow non-AWS users to easily transfer the functionality
to their environment. Hence it currently does not have the functionality to query
status of the job or to terminate the ASG when the job is done (there is nothing
that is monitoring the job!). In practice consider using EKS, Batch, or SageMaker.

### Stop the script
To stop the job and tear down the resources, use the `kill_job` command:

``` bash
python3 petctl.py kill_job --name ${USER}-job
```

You'll notice that the two ASGs created with the `run_job` command are deleted.
