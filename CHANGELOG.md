# CHANGELOG

## 0.2.2 (Feb 18, 2021)

> **_NOTE:_** This is the last release for torchelastic! We are upstreaming TorchElastic into
> pytorch. See [pytorch issue-50621](https://github.com/pytorch/pytorch/issues/50621).

### PyTorch Elastic

* (new) `torchelastic.multiprocessing`, drop in replacement for `torch.multiprocessing` that supports:
   * both function and binary launches
   * inter-process exception propagation
   * piping worker stdout/stderr to separate log files
   * tail worker log files to main console with `{role}_{rank}:` prefix on each line
* Improvements to `torchelastic.events`
* `NCCL_ASYNC_ERROR_HANDLING` set by default in torchelastic agent
* Implemented shutdown barrier on agent to reduce exit time variance
* Minor cosmetic improvements to rendezvous configuration
* Non functional refactoring of `EtcdRendezvous`
* TSM API improvements

## 0.2.1 (October 05, 2020)

### PyTorch Elastic

> **_NOTE:_** As of torch-1.7 and torchelastic-0.2.1 torchelastic will be bundled into the main [pytorch docker](https://hub.docker.com/r/pytorch/pytorch)
   image. [torchelastic/examples](https://hub.docker.com/r/torchelastic/examples) will be available post torch-1.7 release since
   its base image will now be **pytorch/pytorch**

* Torchelastic agent:
  * `run_id` available to workers as `TORCHELASTIC_RUN_ID` environment variable
  * Allow `max_restarts=0`
  * Worker exit barrier added to torchelastic agent to protect against variances in worker finish times
  * Improvements to error handling and propagation from torchelastic agent
  * Enable fault handlers on worker processes to get torch C++ stack traces

* `torchelastic.distributed.launch` CLI:
   * New option `--role` to allow users to set worker role name
   * CLI options can now be set via environment variables (e.g. `PET_NNODES="1:2"`)
   
* Project:
  * Upgraded to Python 3.8
  * Tests moved to `test` directory within the respective modules
  * Use Pyre
  
* Deprecated:
  * [pytorch/elastic](https://hub.docker.com/r/pytorch/elastic) Docker image

* Experimental:
  * [Training Session Manager (TSM)](http://pytorch.org/elastic/0.2.1/tsm_driver.html)  with localhost scheduler
  * [torchelastic.multiprocessing](http://pytorch.org/elastic/0.2.1/multiprocessing.html)


## 0.2.0 (April 29, 2020)

### PyTorch Elastic

* Separate infrastructure related work from the user script. [DesignDoc]
* Events API

[DesignDoc]: https://github.com/pytorch/elastic/blob/master/design/torchelastic/0.2.0/design_doc.md

## 0.1.0rc1 (December 06, 2019)

### PyTorch Elastic

* First release torchelastic v0.1.0rc1 (experimental)
