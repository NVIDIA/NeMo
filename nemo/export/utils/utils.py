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

import os
import shutil

from pathlib import Path


def is_nemo2_checkpoint(checkpoint_path: str) -> bool:
    """
    Checks if the checkpoint is in NeMo 2.0 format.
    Args:
        checkpoint_path (str): Path to a checkpoint.
    Returns:
        bool: True if the path points to a NeMo 2.0 checkpoint; otherwise false.
    """

    ckpt_path = Path(checkpoint_path)
    return (ckpt_path / 'context').is_dir()


def prepare_directory_for_export(model_dir: str, delete_existing_files: bool) -> None:
    """
    Prepares model_dir path for the TRT-LLM/vLLM export.
    Makes sure, that the model_dir directory exists and is empty.

    Args:
        model_dir (str): Path to the target directory for the export.
        delete_existing_files (bool): Attempt to delete existing files if they exist.
    Returns:
        None
    """

    if Path(model_dir).exists():
        if delete_existing_files and len(os.listdir(model_dir)) > 0:
            for files in os.listdir(model_dir):
                path = os.path.join(model_dir, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)

            if len(os.listdir(model_dir)) > 0:
                raise Exception("Couldn't delete all files in the target model directory.")
        elif len(os.listdir(model_dir)) > 0:
            raise Exception("There are files in this folder. Try setting delete_existing_files=True.")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    
