"""
torchelastic.multiprocessing is a wrapper around the native :mod:`multiprocessing`
module. It registers built in or custom error reporter, that enables error capturing
across subprocesses and faciliates error propagation in the parent process

Added helper function to spawn N processes and wait for completion of any of
them. This depends `mp.get_context` which was added in Python 3.4.
"""
# @manual=//caffe2:utils_internal
from torch._utils_internal import get_file_path, prepare_multiprocessing_environment


def _initialize():
    # Setup LD_LIBRARY_PATH for subprocesses
    prepare_multiprocessing_environment(get_file_path("torch"))


_initialize()
