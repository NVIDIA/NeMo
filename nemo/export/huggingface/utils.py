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
from typing import Any, Dict, Tuple, Union

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_distributed_model_weights
from nemo.utils import logging

llm_available = True
try:
    from nemo.collections import llm  # noqa: F401
except ImportError:
    llm = None
    llm_available = False
    logging.warning(
        "nemo.collections.llm module is not available, model exporters will not be connected to llm.GPTModels"
    )


_EXPORTERS_REGISTRY = {}


def get_model(name: str):
    if llm_available:
        model_cls = getattr(llm, name)
        if isinstance(model_cls, type):
            return model_cls

    return type(
        name, (object,), {"__module__": __name__, "__doc__": f"Generated dummy class for missing model: {name}"}
    )


def io_model_exporter(cls, format):
    """
    Wrapper around `nemo.lightning.io.model_exporter` that (1) registers the
    exporter with the NeMo IO system and (2) stores a reference to the exporter
    class in the local `_EXPORTERS_REGISTRY` so it can be retrieved through
    `get_exporter`.
    """
    if not llm_available:

        def noop(exporter_cls, *args, **kwargs):
            key = cls if isinstance(cls, str) else getattr(cls, "__name__", str(cls))
            if format not in _EXPORTERS_REGISTRY:
                _EXPORTERS_REGISTRY[format] = {}
            _EXPORTERS_REGISTRY[format][key] = exporter_cls
            return exporter_cls

        return noop

    from nemo.lightning.io import model_exporter as _model_exporter

    base_decorator = _model_exporter(cls, format)

    def decorator(exporter_cls):
        decorated_cls = base_decorator(exporter_cls)
        key = cls if isinstance(cls, str) else getattr(cls, "__name__", str(cls))
        if format not in _EXPORTERS_REGISTRY:
            _EXPORTERS_REGISTRY[format] = {}
        _EXPORTERS_REGISTRY[format][key] = decorated_cls
        return decorated_cls

    return decorator


def get_exporter(cls, format):
    key = cls if isinstance(cls, str) else getattr(cls, "__name__", str(cls))
    if format not in _EXPORTERS_REGISTRY:
        return None
    return _EXPORTERS_REGISTRY[format].get(key, None)


def torch_dtype_from_mcore_config(config) -> torch.dtype:
    """Extract the appropriate torch dtype from a Megatron Core configuration.

    Args:
        config: Megatron Core Transformer configuration

    Returns:
        torch.dtype: The appropriate torch dtype (float16, bfloat16, or float32)
    """
    if config.fp16:
        return torch.float16
    elif config.bf16:
        return torch.bfloat16
    else:
        return torch.float


def change_paths_to_absolute_paths(tokenizer_config: Dict[Any, Any], nemo_checkpoint: Path) -> Dict[Any, Any]:
    """
    Creates absolute path to the local tokenizers. Used for NeMo 2.0.

    Args:
        tokenizer_config (dict): Parameters for instantiating the tokenizer.
        nemo_checkpoint (path): Path to the NeMo2 checkpoint.
    Returns:
        dict: Updated tokenizer config.
    """
    context_path = nemo_checkpoint / 'context'

    # 'pretrained_model_name' -- huggingface tokenizer case
    # 'model_path' -- sentencepiece tokenizer
    path_keys = ['pretrained_model_name', 'model_path']

    for path_key in path_keys:
        if path := tokenizer_config.get(path_key, None):
            tokenizer_path = context_path / path
            if not tokenizer_path.exists():
                continue

            tokenizer_config[path_key] = str(tokenizer_path.resolve())

    return tokenizer_config


def load_config(path: Union[str, Path]):
    """Load the config from the NeMo model.

    Args:
        path (str | Path): The path to the NeMo model.

    Returns:
        Config: The config from the NeMo model.
    """
    model_yaml = Path(path) / "context" / "model.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError("model.yaml is not found in the context folder of the checkpoint.")
    with open(model_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
    dict_to_obj = lambda d: (
        type('Config', (), {kk: dict_to_obj(vv) for kk, vv in d.items()}) if isinstance(d, dict) else d
    )

    return dict_to_obj(config['config'])


def ckpt_load(checkpoint_path: str) -> Tuple[Dict, Any]:
    """
    This function loads the state dict directly from a distributed checkpoint, and modify the state dict
    so that it is consistent with the key names you would get from loading the checkpoint into a model.
    This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

    Args:
        path (Path): The path from which the model will be loaded.

    Returns
    -------
        Tuple[Dict, Any]: The loaded state dict and the yaml config object.
    """
    config_obj = load_config(checkpoint_path)

    path = Path(checkpoint_path)
    langauge_layers = config_obj.num_layers

    distributed_model_weights = load_distributed_model_weights(path, True).items()

    state_dict = {}
    for k, v in distributed_model_weights:
        if '_extra_state' in k:
            continue
        new_k = k.replace("module.", "")
        if 'layers' in new_k and v.size(0) == langauge_layers:
            # Only split layers
            for i in range(v.size(0)):
                state_dict[new_k.replace('layers', f'layers.{str(i)}')] = v[i]
        state_dict[new_k] = v

    return state_dict, config_obj


def get_tokenizer(path: str) -> "TokenizerSpec":
    """Get the tokenizer from the NeMo model.

    Returns:
        TokenizerSpec: Tokenizer from the NeMo model
    """

    nemo_checkpoint = Path(path)
    tokenizer_config = OmegaConf.load(nemo_checkpoint / "context/model.yaml").tokenizer
    if ('additional_special_tokens' in tokenizer_config) and len(tokenizer_config['additional_special_tokens']) == 0:
        del tokenizer_config['additional_special_tokens']

    tokenizer_config = change_paths_to_absolute_paths(tokenizer_config, nemo_checkpoint)
    tokenizer = instantiate(tokenizer_config)

    if hasattr(tokenizer, 'bos_id'):
        tokenizer.tokenizer.bos_token_id = tokenizer.bos_id
    if hasattr(tokenizer, 'eos_id'):
        tokenizer.tokenizer.eos_token_id = tokenizer.eos_id

    return tokenizer


def load_connector(path, target):
    model_yaml = Path(str(path)) / "context" / "model.yaml"
    with open(model_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
    
    model_class = config['_target_'].split('.')[-1]
    exporter = get_exporter(model_class, target)
    if exporter is None:
        raise ValueError(f"Unsupported model type: {model_class}")
    return exporter(Path(path))