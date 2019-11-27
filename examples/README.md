# Examples

The examples in this directory can be run on AWS using `petctl` (see 
[usage](../aws/README.md)).

> The sections below assume that you have run `petctl.py configure`.
## Classy Vision

[Classy Vision](https://classyvision.ai) is a PyTorch framework for image and video classification. It
can be run on torchelastic to make the runs fault tolerant. To invoke a classy
run on the local host (GPU required), run:

```
cd $torchelastic_git_repo_root
python3 examples/classy_vision/main.py \
  --config_file configs/resnet50_synthetic_image_classy_config.json \
  --checkpoint_folder ~/classy_vision/checkpoint
```

To run on a larger training job on multiple instances on AWS use:

```
cd $torchelastic_git_repo_root
export JOB_SIZE=4
export JOB_NAME=my_job

python3 aws/petctl.py run_job \
   --size $JOB_SIZE \
   --name $JOB_NAME \
   examples/classy_vision/main.py \
   -- \
   --config_file configs/resnet50_synthetic_image_classy_config.json \
   --checkpoint_folder /mnt/efs/fs1/classy_vision/checkpoint
```

## Imagenet

Running the image net example is similar to running the classy vision example.

```
cd $torchelastic_git_repo_root
export JOB_SIZE=4
export JOB_NAME=my_job
export SPECS=~/your_specs_file.json

python3 aws/petctl.py run_job \
   --size $JOB_SIZE \
   --name $JOB_NAME \
   examples/imagenet/main.py \
   -- \
   --epochs 90 \
   --input_path /mnt/efs/fs1/data/imagenet/train
```