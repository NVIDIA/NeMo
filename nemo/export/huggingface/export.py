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

from pathlib import Path
import yaml

from nemo.export.huggingface.utils import get_exporter, export_ckpt

def load_connector(path, target):
    """
    Load the connector for the given model and target.
    """
    model_yaml = Path(str(path)) / "context" / "model.yaml"
    with open(model_yaml, 'r') as stream:
        config = yaml.safe_load(stream)

    model_class = config['_target_'].split('.')[-1]
    exporter = get_exporter(model_class, target)
    if exporter is None:
        raise ValueError(f"Unsupported model type: {model_class}")
    return exporter(Path(path))


def export_to_hf(model_path, model_dir):
    """
    Export the model to the Hugging Face format.

    Args:
        model_path (str): The path to the model.
        model_dir (str): The directory to save the model.
    """
    return export_ckpt(
        path=model_path, target='hf', output_path=Path(model_dir), overwrite=True, load_connector=load_connector
    )
