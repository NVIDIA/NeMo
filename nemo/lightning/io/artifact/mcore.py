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

"""HuggingFace model serialization support for NeMo's configuration system.

This module provides integration between NeMo's configuration system and HuggingFace's
pretrained models. It enables automatic serialization and deserialization of HuggingFace
models within NeMo's configuration framework.

The integration works by:
1. Detecting HuggingFace models through their characteristic methods (save_pretrained/from_pretrained)
2. Converting them to Fiddle configurations that preserve the model's class and path
3. Providing an artifact handler (HFAutoArtifact) that manages the actual model files

Example:
    ```python
    from transformers import AutoModel
    
    # This model will be automatically handled by the HFAutoArtifact system
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    # When serialized, the model files will be saved to the artifacts directory
    # When deserialized, the model will be loaded from the saved files
    ```
"""

import shutil

from nemo.lightning.io.artifact import Artifact, DirOrStringArtifact, FileArtifact
from nemo.lightning.io.mixin import track_io
from pathlib import Path
from typing import Union

import os
import fiddle as fdl


class MCoreArtifact(Artifact):
    def dump(
        self,
        instance,
        path: str,
        absolute_dir: Path,
        relative_dir: Path,
    ) -> str:
        value = pathize(path)
        if instance.library == 'huggingface':
            if value.exists():
                # if HF tokenizer is stored locally
                relative_dir = pathize(relative_dir) / pathize(value.name)
                os.makedirs(str(absolute_dir / relative_dir), exist_ok=True)
                for file in value.iterdir():
                    copy_file(file, absolute_dir, relative_dir)
                return str(relative_dir)
            else:
                # if HF tokenizer is loaded from HF cloud
                path_to_save = pathize(absolute_dir) / pathize(value.name)
                instance.save_pretrained(path_to_save)
                copy_file(instance.metadata_path, path_to_save, relative_dir)
                return str(path_to_save.name)
        else:
            # save tokenizer and it's metadata for SentencePiece and TikToken
            copy_file(instance.metadata_path, absolute_dir, relative_dir)
            new_value = copy_file(value, absolute_dir, relative_dir)
            return str(new_value)

    def load(self, path: str) -> str:
        return path


def copy_file(src: Union[Path, str], path: Union[Path, str], relative_dst: Union[Path, str]):
    relative_path = pathize(relative_dst) / pathize(src).name
    output = pathize(path) / relative_path
    if output.exists():
        raise FileExistsError(f"Dst file already exists {str(output)}")
    shutil.copy2(src, output)
    return relative_path


def pathize(s):
    if not isinstance(s, Path):
        return Path(s)
    return s