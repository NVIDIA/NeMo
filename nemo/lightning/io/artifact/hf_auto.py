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

import inspect
from pathlib import Path

import fiddle as fdl

from nemo.lightning.io.artifact import Artifact
from nemo.lightning.io.to_config import to_config


class HFAutoArtifact(Artifact):
    def dump(self, instance, value: Path, absolute_dir: Path, relative_dir: Path) -> Path:
        instance.save_pretrained(Path(absolute_dir) / "artifacts")

        return "./" + str(Path(relative_dir) / "artifacts")

    def load(self, path: Path) -> Path:
        return path


def from_pretrained(auto_cls, pretrained_model_name_or_path="dummy"):
    return auto_cls.from_pretrained(pretrained_model_name_or_path)


@to_config.register(
    lambda v: not inspect.isclass(v)
    and getattr(v, "__module__", "").startswith("transformers")
    and hasattr(v, "save_pretrained")
    and hasattr(v, "from_pretrained")
)
def handle_hf_pretrained(value):
    return fdl.Config(
        from_pretrained,
        auto_cls=value.__class__,
        pretrained_model_name_or_path="dummy",
    )


from_pretrained.__io_artifacts__ = [HFAutoArtifact("pretrained_model_name_or_path")]
