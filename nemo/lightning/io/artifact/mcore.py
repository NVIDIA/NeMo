# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
MegatronCore tokenizer serialization support for NeMo's configuration system.

This module provides integration between NeMo's configuration system and MegatronCore's
tokenizers. It enables automatic serialization and deserialization of MegatronCore tokenizers
within NeMo's configuration framework.
"""

import os
import shutil
from pathlib import Path
from typing import Union

from nemo.lightning.io.artifact import Artifact


class MCoreArtifact(Artifact):
    def dump(
        self,
        instance,
        path: str,
        absolute_dir: Path,
        relative_dir: Path,
        dir_name: str = "tokenizer",
    ) -> str:
        """Saves MegatronCore tokenizer to checkpoint directory."""

        path = pathize(path)
        path_to_save = pathize(absolute_dir) / pathize(dir_name)
        if instance.library == 'huggingface':
            if path.exists():
                # if HF tokenizer is stored locally
                os.makedirs(str(path_to_save), exist_ok=True)
                for file in path.iterdir():
                    copy_file(file, path_to_save, relative_dir)
            else:
                # if HF tokenizer is loaded from HF cloud
                path_to_save = pathize(absolute_dir) / pathize(dir_name)
                instance.save_pretrained(path_to_save)
                copy_file(instance.metadata_path, path_to_save, relative_dir)
            vocab_file, merge_file = instance.vocab_file, instance.merge_file
            if vocab_file:
                copy_file(pathize(vocab_file), path_to_save, relative_dir, overwrite=True)
            if merge_file:
                copy_file(pathize(merge_file), path_to_save, relative_dir, overwrite=True)
            return dir_name
        else:
            # save tokenizer and it's metadata for SentencePiece and TikToken
            os.makedirs(str(path_to_save), exist_ok=True)
            new_path = copy_file(path, path_to_save, relative_dir)
            copy_file(instance.metadata_path, path_to_save, relative_dir)
            return f"{dir_name}/{str(new_path)}"

    def load(self, path: str) -> str:
        return path


def copy_file(
        src: Union[Path, str],
        path: Union[Path, str],
        relative_dst: Union[Path, str],
        overwrite: bool = False,
    ) -> Path:
    """
    Copies files to checkpoint directory.

    Args:
        src (Union[Path, str]): path to the file to be copied.
        path (Union[Path, str]): path where to save copied file.
        relative_dst (Union[Path, str]): name of the copied file.
        overwrite (bool): whether to overwrite the file if it exists.

    Returns:
        Path: Path objecy of copied file.
    """

    relative_path = pathize(relative_dst) / pathize(src).name
    output = pathize(path) / relative_path
    if output.exists() and not overwrite:
        raise FileExistsError(f"Dst file already exists {str(output)}")
    shutil.copy2(src, output)
    return relative_path


def pathize(path: Union[str, Path]) -> Path:
    """
    Converts str path to Path object.

    Args:
        path (str): path to the file.
    
    Retunrs:
        Path: file's Path object.
    """

    if not isinstance(path, Path):
        return Path(path)
    return path