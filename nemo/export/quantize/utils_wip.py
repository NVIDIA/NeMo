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


def save_artifacts(model, output_dir: str, use_abspath: bool = False) -> None:
    """Save all model artifacts and tokenizer config to a given output directory."""
    app_state = AppState()
    model_file = app_state.model_restore_path
    model_cfg = copy.deepcopy(model.cfg)

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
            arti_file = arti_item.path.removeprefix("nemo:")
            arti_path = os.path.join(output_dir, arti_name)
            if maybe_tar is not None:
                maybe_tar.extract(f"./{arti_file}", path=output_dir)
                os.rename(os.path.join(output_dir, arti_file), arti_path)
            else:
                shutil.copy(os.path.join(model_file, arti_file), arti_path)
            # Store artifact path as basename by default. Otherwise save absolute path but bear in mind
            # that in this case output directory should be permanent for correct artifact recovery later
            arti_path = os.path.abspath(arti_path) if use_abspath else os.path.basename(arti_path)
            OmegaConf.update(model_cfg, arti_name, arti_path)
    OmegaConf.save(model_cfg.tokenizer, os.path.join(output_dir, "tokenizer_config.yaml"))
