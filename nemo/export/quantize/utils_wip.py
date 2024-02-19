# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import copy
import os
import shutil
import tarfile
import tempfile

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from nemo.utils.app_state import AppState
from nemo.utils.get_rank import get_rank, is_global_rank_zero


@contextlib.contextmanager
def temporary_directory():
    """Create a shared temporary directory across ranks in distributed setup.

    This function assumes that the distributed setup has been already
    correctly initialized. It is intended to be used only in single-node
    setup so that all ranks can access the directory created."""

    if is_global_rank_zero():
        tmp_dir = [tempfile.TemporaryDirectory()]
    else:
        tmp_dir = [None]
    torch.distributed.broadcast_object_list(tmp_dir)
    print(f"[{get_rank()}] tmp_dir={tmp_dir}")  # TODO: remove debug print
    yield tmp_dir[0].name
    # We use barrier below to make sure that rank zero won't exit
    # and delete tmp_dir while other ranks may still use it
    dist.barrier()


def copy_artifacts(model, output_dir: str):
    """Copy all model artifacts to a given output directory and return modified config."""
    app_state = AppState()
    model_file = app_state.model_restore_path
    model_config = copy.deepcopy(model.cfg)

    # Setup model file handling context: directory or tarball
    if os.path.isfile(model_file):
        model_file_handler = tarfile.open
        kwargs = {"name": model_file, "mode": "r:"}
    elif os.path.isdir(model_file):
        model_file_handler = contextlib.nullcontext
        kwargs = {}
    else:
        raise FileNotFoundError(model_file)

    # Copy or extract artifacts depending on the context
    with model_file_handler(**kwargs) as maybe_tar:
        for arti_name, arti_item in model.artifacts.items():
            _, arti_file = arti_item.path.split("nemo:")
            if maybe_tar is not None:
                maybe_tar.extract(f"./{arti_file}", path=output_dir)
            else:
                shutil.copy(os.path.join(model_file, arti_file), output_dir)
            # Update artifact path to basename
            OmegaConf.update(model_config, arti_name, os.path.basename(arti_file))
    return model_config
