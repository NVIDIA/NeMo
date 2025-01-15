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
    """Artifact handler for HuggingFace pretrained model/processor/tokenizer/etc..

    This handler manages the serialization and deserialization of HuggingFace models
    by utilizing their save_pretrained/from_pretrained methods. It saves models to
    an 'artifacts' subdirectory within the specified path.
    """

    def dump(self, instance, value: Path, absolute_dir: Path, relative_dir: Path) -> Path:
        """Save a HuggingFace model to disk.

        Args:
            instance: The HuggingFace model instance to save
            value: Original path value (unused)
            absolute_dir: Absolute path to the save directory
            relative_dir: Relative path from the config file to the save directory

        Returns:
            str: The relative path to the saved model artifacts
        """
        instance.save_pretrained(Path(absolute_dir) / "artifacts")
        return "./" + str(Path(relative_dir) / "artifacts")

    def load(self, path: Path) -> Path:
        """Return the path to load a HuggingFace model.

        Args:
            path: Path to the saved model artifacts

        Returns:
            Path: The same path, to be used with from_pretrained
        """
        return path


def from_pretrained(auto_cls, pretrained_model_name_or_path="dummy", trust_remote_code=False):
    """Factory function for loading HuggingFace pretrained models.

    This function is used as the serialization target for HuggingFace models.
    When deserialized, it will recreate the model using its from_pretrained method.

    Args:
        auto_cls: The HuggingFace model class (e.g., AutoModel, AutoTokenizer)
        pretrained_model_name_or_path: Path to the saved model or model identifier
        trust_remote_code: If True, allows execution of custom model code from the Hub. Use only for trusted repositories. Default is False

    Returns:
        The loaded HuggingFace model
    """
    return auto_cls.from_pretrained(pretrained_model_name_or_path, trust_remote_code)


@to_config.register(
    lambda v: not inspect.isclass(v)
    and getattr(v, "__module__", "").startswith("transformers")
    and hasattr(v, "save_pretrained")
    and hasattr(v, "from_pretrained")
)
def handle_hf_pretrained(value):
    """Convert a HuggingFace model instance to a Fiddle configuration.

    This handler detects HuggingFace model instances by checking for the presence
    of save_pretrained and from_pretrained methods. It converts them to a Fiddle
    configuration that will recreate the model using from_pretrained.

    Args:
        value: A HuggingFace model instance

    Returns:
        fdl.Config: A Fiddle configuration that will recreate the model
    """
    return fdl.Config(
        from_pretrained,
        auto_cls=value.__class__,
        pretrained_model_name_or_path="dummy",
    )


# Register the HFAutoArtifact handler for the pretrained_model_name_or_path parameter
from_pretrained.__io_artifacts__ = [HFAutoArtifact("pretrained_model_name_or_path")]
