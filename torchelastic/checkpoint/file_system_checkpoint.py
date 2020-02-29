#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from .api import Checkpoint, CheckpointManager


log = logging.getLogger(__name__)


class FileSystemCheckpointManager(CheckpointManager):
    """
    A CheckpointManager that reads/writes checkpoints to
    the file system.
    """

    def __init__(self, checkpoint_dir):
        self._checkpoint_dir = checkpoint_dir

    def _get_sequence_ids(self):
        # r=root, d=directories, f = files
        def _is_int(input):
            try:
                int(input)
            except ValueError:
                return False
            return True

        dirs = []
        for entry_name in os.listdir(self._checkpoint_dir):
            entry_path = os.path.join(self._checkpoint_dir, entry_name)
            if os.path.isdir(entry_path):
                dirs.append(entry_name)

        seq_ids = [int(id) for id in dirs if _is_int(id)]
        # sort desc
        seq_ids.sort(reverse=True)
        return seq_ids

    def create_checkpoint(self):
        """
        create a new checkpoint
        """
        seq_ids = self._get_sequence_ids()
        next_id = 0 if len(seq_ids) == 0 else seq_ids[0] + 1
        return FileSystemCheckpoint(next_id, self._checkpoint_dir)

    def get_checkpoint(self, sequence_id):
        current_folder = os.path.join(self._checkpoint_dir, str(sequence_id))
        if not os.path.isdir(current_folder):
            raise Exception("folder: {} not found".format(current_folder))
        return FileSystemCheckpoint(sequence_id, self._checkpoint_dir)

    def get_latest_checkpoint(self):
        seq_ids = self._get_sequence_ids()
        if len(seq_ids) == 0:
            return None
        else:
            return self.get_checkpoint(seq_ids[0])

    def list_checkpoints(self):
        seq_ids = self._get_sequence_ids()
        return [self.get_checkpoint(id) for id in seq_ids]


class FileSystemCheckpoint(Checkpoint):
    """
    Represents a checkpoint in the local file system.
    """

    def __init__(self, sequence_id, checkpoint_dir):
        self.checkpoint_dir = os.path.join(checkpoint_dir, str(sequence_id))
        # Create target Directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
            log.info(f"Created checkpoint dir: {self.checkpoint_dir}")

    def open_output_stream(self, key):
        path = os.path.join(self.checkpoint_dir, key)
        log.info(f"saving checkpoint to: {path}")
        return open(path, "wb")

    def open_input_stream(self, key):
        path = os.path.join(self.checkpoint_dir, key)
        log.info("Loading checkpoint from: {}".format(path))
        return open(path, "rb")

    def commit(self):
        pass

    def discard(self):
        pass
