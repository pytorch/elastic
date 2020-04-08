[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)[![CircleCI](https://circleci.com/gh/pytorch/elastic.svg?style=svg&circle-token=9bea46e94adbe2f3e0fb2d4054b1b655f2e208c2)](https://circleci.com/gh/pytorch/elastic)

# TorchElastic

TorchElastic allows you to launch distributed PyTorch jobs in a
fault-tolerant and elastic manner.
For the latest documentation, please refer to our
[website](https://pytorch.org/elastic).


## Requirements
torchelastic requires
* python3 (3.6+)
* torch
* etcd

## Installation
```bash
pip install torchelastic
```

## Quickstart

**Fault-tolerant** on `4` nodes, `8` trainers/node, total `4 * 8 = 32` trainers.
Run the following on all nodes.
```bash
python -m torchelastic.distributed.launch
            --nnodes=4
            --nproc_per_node=8
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

**Elastic on** `1 ~ 4` nodes, `8` trainers/node, total `8 ~ 32` trainers. Job
starts as soon as `1` node is healthy, you may add up to `4` nodes.
```bash
python -m torchelastic.distributed.launch
            --nnodes=1:4
            --nproc_per_node=8
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

```
## Contributing

We welcome PRs. See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License
torchelastic is BSD licensed, as found in the [LICENSE](LICENSE) file.
