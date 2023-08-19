import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, get_args, get_origin

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import is_gated_activation
from transformers import GPT2Config
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .tensor_utils import get_tensor_from_file, split, torch_to_numpy_with_dtype

DECODER_GPT2 = "gpt2"
DECODER_GPTJ = "gptj"
DECODER_LLAMA = "llama"
DECODER_GPTNEXT = "gptnext"

QUANTIZATION_NONE = ""
QUANTIZATION_FP8 = "fp8"
QUANTIZATION_INT8_SQ = "int8_sq"

LINEAR_COLUMN = "column"
LINEAR_ROW = "row"

LAYERNORM_DEFAULT = ""
LAYERNORM_RMS = "rms"


@dataclass
class EmbeddingConfig:
    """The embedding layer config."""

    weight: np.array = None

    @staticmethod
    def from_nn_module(module: nn.Module, dtype=trt.float16):
        """Converts an nn.Module to an EmbeddingConfig."""
        return EmbeddingConfig(weight=torch_to_numpy_with_dtype(module.weight, dtype))

    @property
    def vocab_size(self):
        """Infers the vocab_size from the embedding layer weights shape."""
        return self.weight.shape[0]

    @property
    def hidden_size(self):
        """Infers the hidden_size from the embedding layer weights shape."""
        return self.weight.shape[1]


@dataclass
class LayernormConfig:
    """The layernorm layer config."""

    weight: np.array = None
    bias: np.array = None
    layernorm_type: str = LAYERNORM_DEFAULT

    @staticmethod
    def from_nn_module(module: nn.Module, dtype=trt.float16):
        """Converts an nn.Module to an LayernormConfig."""
        layernorm_type = LAYERNORM_RMS if type(module) == LlamaRMSNorm else LAYERNORM_DEFAULT

        config = LayernormConfig(weight=torch_to_numpy_with_dtype(module.weight, dtype), layernorm_type=layernorm_type)
        if layernorm_type == LAYERNORM_DEFAULT:
            config.bias = torch_to_numpy_with_dtype(module.bias, dtype)

        return config


@dataclass
class LinearConfig:
    """The linear layer config."""

    linear_type: str = ""
    weight: np.array = None
    bias: np.array = None
    activation_scaling_factor: np.array = None
    weights_scaling_factor: np.array = None
    prequant_scaling_factor: np.array = None

    @staticmethod
    def from_nn_module(module: nn.Module, linear_type: str, rank=0, tensor_parallel=1, dtype=trt.float16):
        """Converts an nn.Module to an LinearConfig."""
        weight = torch_to_numpy_with_dtype(module.weight, dtype)
        if "Conv1D" in type(module).__name__:
            weight = weight.transpose()
        else:
            assert type(module) == nn.Linear

        config = LinearConfig()
        config.linear_type = linear_type
        config.weight = np.ascontiguousarray(
            split(weight, tensor_parallel, rank, dim=0 if linear_type == LINEAR_COLUMN else 1)
        )

        if hasattr(module, "bias") and module.bias is not None:
            if linear_type == LINEAR_COLUMN:
                config.bias = np.ascontiguousarray(
                    split(
                        torch_to_numpy_with_dtype(module.bias, dtype),
                        tensor_parallel,
                        rank,
                    )
                )
            else:
                config.bias = torch_to_numpy_with_dtype(module.bias, dtype)

        return config

    @staticmethod
    def from_qkv_nn_modules(qkv_modules: List[nn.Module], rank=0, tensor_parallel=1, dtype=trt.float16):
        """Converts the qkv modules to an LinearConfig."""
        config = LinearConfig()
        config.linear_type = LINEAR_COLUMN
        if len(qkv_modules) == 1:
            # QKV layers combined as a single module, e.g. GPT2
            qkv_module = qkv_modules[0]
            assert "Conv1D" in type(qkv_module).__name__

            qkv_shape = qkv_module.weight.shape
            # Decode the concat QKV weights and split them to different GPU rank.
            config.weight = np.ascontiguousarray(
                split(
                    torch_to_numpy_with_dtype(qkv_module.weight, dtype=dtype).reshape(
                        qkv_shape[0], 3, qkv_shape[-1] // 3
                    ),
                    tensor_parallel,
                    rank,
                    dim=-1,
                )
                .reshape(qkv_shape[0], -1)
                .transpose()
            )
            config.bias = np.ascontiguousarray(
                split(
                    torch_to_numpy_with_dtype(qkv_module.bias, dtype=dtype).reshape(3, qkv_shape[-1] // 3),
                    tensor_parallel,
                    rank,
                    dim=-1,
                ).reshape(-1)
            )

        elif len(qkv_modules) == 3:
            # Separate QKV layers
            for m in qkv_modules:
                assert type(m) == nn.Linear
                assert not (hasattr(m, "bias") and m.bias is not None)

            q_weight = qkv_modules[0].weight
            k_weight = qkv_modules[1].weight
            v_weight = qkv_modules[2].weight

            qkv_weight = torch_to_numpy_with_dtype(torch.stack([q_weight, k_weight, v_weight]), dtype)

            q_emb = qkv_weight.shape[1]
            model_emb = qkv_weight.shape[2]
            split_v = split(qkv_weight, tensor_parallel, rank, dim=1)
            split_v = split_v.reshape(3 * (q_emb // tensor_parallel), model_emb)
            config.weight = np.ascontiguousarray(split_v)

        else:
            assert False, f"QKV modules format {qkv_modules} not supported"

        return config


@dataclass
class AttentionConfig:
    """The attention layer config."""

    qkv: LinearConfig = None
    dense: LinearConfig = None

    rotary_dim: int = -np.inf

    @staticmethod
    def from_nemo(
        weights_dir: Path,
        gpt_config: GPT2Config,
        layer_id: int,
        rank: int = 0,
        tensor_parallel: int = 1,
        dtype: trt.DataType = trt.bfloat16,
    ):
        """Converts the nemo weights and config to `AttentionConfig`."""

        attention = AttentionConfig()
        attention.qkv = LinearConfig(linear_type=LINEAR_COLUMN)
        n_embd = gpt_config.n_embd
        c_attn_out_dim = 3 * n_embd // tensor_parallel
        attention.qkv.weight = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    weights_dir,
                    f"layers.{layer_id}.attention.query_key_value.weight.{rank}",
                    shape=[n_embd, c_attn_out_dim],
                    dtype=dtype,
                ),
                [1, 0],
            )
        )
        attention.qkv.bias = get_tensor_from_file(
            weights_dir,
            f"layers.{layer_id}.attention.query_key_value.bias.{rank}",
            dtype=dtype,
        )

        attention.dense = LinearConfig(linear_type=LINEAR_ROW)
        attention.dense.weight = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    weights_dir,
                    f"layers.{layer_id}.attention.dense.weight.{rank}",
                    shape=[n_embd // tensor_parallel, n_embd],
                    dtype=dtype,
                ),
                [1, 0],
            )
        )
        attention.dense.bias = get_tensor_from_file(
            weights_dir,
            f"layers.{layer_id}.attention.dense.bias",
            dtype=dtype,
        )
        return attention


@dataclass
class MLPConfig:
    """The MLP layer config."""

    fc: LinearConfig = None
    gate: LinearConfig = None
    proj: LinearConfig = None
    hidden_act: str = ""

    @staticmethod
    def from_nemo(
        weights_dir: Path,
        gpt_config: GPT2Config,
        layer_id: int,
        rank: int = 0,
        tensor_parallel: int = 1,
        dtype: trt.DataType = trt.bfloat16,
    ):
        """Converts the nemo weights and config to `MLPConfig`."""
        n_embd = gpt_config.n_embd
        inter_size = gpt_config.intermediate_size

        mlp = MLPConfig(hidden_act=gpt_config.activation_function)
        mlp.fc = LinearConfig(linear_type=LINEAR_COLUMN)
        mlp.fc.weight = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    weights_dir,
                    f"layers.{layer_id}.mlp.dense_h_to_4h.weight.{rank}",
                    shape=[n_embd, inter_size // tensor_parallel],
                    dtype=dtype,
                ),
                [1, 0],
            )
        )
        mlp.fc.bias = get_tensor_from_file(
            weights_dir,
            f"layers.{layer_id}.mlp.dense_h_to_4h.bias.{rank}",
            dtype=dtype,
        )

        gated = is_gated_activation(mlp.hidden_act)
        if gated:
            mlp.gate = LinearConfig(linear_type=LINEAR_COLUMN)
            mlp.gate.weight = np.ascontiguousarray(
                np.transpose(
                    get_tensor_from_file(
                        weights_dir,
                        f"layers.{layer_id}.mlp.dense_h_to_4h.gate.weight.{rank}",
                        shape=[n_embd, inter_size // tensor_parallel],
                        dtype=dtype,
                    ),
                    [1, 0],
                )
            )
            mlp.gate.bias = get_tensor_from_file(
                weights_dir,
                f"layers.{layer_id}.mlp.dense_h_to_4h.gate.bias.{rank}",
                dtype=dtype,
            )

        mlp.proj = LinearConfig(linear_type=LINEAR_ROW)
        mlp.proj.weight = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    weights_dir,
                    f"layers.{layer_id}.mlp.dense_4h_to_h.weight.{rank}",
                    shape=[inter_size // tensor_parallel, n_embd],
                    dtype=dtype,
                ),
                [1, 0],
            )
        )
        mlp.proj.bias = get_tensor_from_file(weights_dir, f"layers.{layer_id}.mlp.dense_4h_to_h.bias", dtype=dtype)
        return mlp


@dataclass
class DecoderLayerConfig:
    """The decoder layer config."""

    decoder_type: str = ""
    input_layernorm: LayernormConfig = None
    attention: AttentionConfig = None
    post_layernorm: LayernormConfig = None
    mlp: MLPConfig = None

    num_attention_heads: int = 0
    max_position_embeddings: int = 0
    rotary_pct: float = 0

    @property
    def hidden_size(self):
        return self.mlp.fc.weight.shape[1]

    @property
    def ffn_hidden_size_local(self):
        return self.mlp.fc.weight.shape[0]

    @staticmethod
    def from_nemo(
        weights_dir: Path,
        gpt_config: GPT2Config,
        decoder_type: str,
        layer_id: int,
        rank: int = 0,
        tensor_parallel: int = 1,
        dtype: trt.DataType = trt.bfloat16,
    ):
        """Converts the nemo weights and config to `DecoderLayerConfig`."""
        layer_config = DecoderLayerConfig(
            decoder_type=decoder_type,
            num_attention_heads=gpt_config.n_head,
            max_position_embeddings=gpt_config.n_positions,
            rotary_pct=gpt_config.rotary_pct,
        )
        layer_config.input_layernorm = LayernormConfig()
        layer_config.input_layernorm.weight = get_tensor_from_file(
            weights_dir,
            f"layers.{layer_id}.input_layernorm.weight",
            dtype=dtype,
        )
        layer_config.input_layernorm.bias = get_tensor_from_file(
            weights_dir,
            f"layers.{layer_id}.input_layernorm.bias",
            dtype=dtype,
        )
        layer_config.post_layernorm = LayernormConfig()
        layer_config.post_layernorm.weight = get_tensor_from_file(
            weights_dir,
            f"layers.{layer_id}.post_attention_layernorm.weight",
            dtype=dtype,
        )
        layer_config.post_layernorm.bias = get_tensor_from_file(
            weights_dir,
            f"layers.{layer_id}.post_attention_layernorm.bias",
            dtype=dtype,
        )

        layer_config.attention = AttentionConfig.from_nemo(
            weights_dir, gpt_config, layer_id, rank, tensor_parallel, dtype
        )
        layer_config.mlp = MLPConfig.from_nemo(weights_dir, gpt_config, layer_id, rank, tensor_parallel, dtype)

        return layer_config


def _from_dict(class_type, data):
    """Helper function to load the data as a class_type. class_type must be a dataclass."""
    if data is None:
        return None

    if dataclasses.is_dataclass(class_type):
        fieldtypes = {f.name: f.type for f in dataclasses.fields(class_type)}
        return class_type(**{f: _from_dict(fieldtypes[f], data[f]) for f in data})
    elif get_origin(class_type) == list and dataclasses.is_dataclass(get_args(class_type)[0]):
        list_value = []
        for child in data:
            child_class_type = get_args(class_type)[0]
            list_value.append(_from_dict(child_class_type, child))
        return list_value
    else:
        return data


@dataclass
class ModelConfig:
    """The full LLM model config that includes the full information needed for tensorrt_llm engine building.

    This class includes all the fields that tensorrt_llm supports, but not all of the fields are required.
    """

    # Global metadata
    quantization: str = QUANTIZATION_NONE
    dtype: str = "float16"

    # Parallel metadata
    rank: int = 0
    tensor_parallel: int = 1

    # Model structure and weights
    vocab_embedding: EmbeddingConfig = None
    positional_embedding: EmbeddingConfig = None
    layers: List[DecoderLayerConfig] = field(default_factory=list)
    final_layernorm: LayernormConfig = None
    lm_head: LinearConfig = None

    def to_dict(self) -> dict:
        """Converts the instance to a python dict"""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict):
        """Load a dict to a `ModelConfig` instance."""
        return _from_dict(ModelConfig, d)

    @property
    def vocab_size(self):
        """Returns the vocab_size of the model."""
        return self.vocab_embedding.vocab_size

    @property
    def vocab_size_padded(self):
        """Returns the padded vocab_size of the model rounds to the tensor_parallel."""
        return pad_vocab_size(self.vocab_size, self.tensor_parallel)

    @property
    def hidden_size(self):
        """Returns the hidden_size of the model."""
        return self.vocab_embedding.hidden_size

    @property
    def max_position_embeddings(self):
        """Returns the max_position_embedding of the model."""
        return self.layers[0].max_position_embeddings

    @property
    def num_attention_heads(self):
        """Returns the num_attention_heads of the model."""
        return self.layers[0].num_attention_heads

    @property
    def hidden_act(self):
        """Returns the hidden_act of the model."""
        return self.layers[0].mlp.hidden_act


def _restore_model_config(model_config, weights):
    """Recursively restores the model_config from json and loads np.ndarray weights from weights."""
    if isinstance(model_config, dict):
        for k, v in model_config.items():
            if isinstance(v, str) and v.startswith("_np:"):
                model_config[k] = weights[v]
            else:
                _restore_model_config(v, weights)
    if isinstance(model_config, list):
        for i, v in enumerate(model_config):
            if isinstance(v, str) and v.startswith("_np:"):
                model_config[i] = weights[v]
            else:
                _restore_model_config(v, weights)


def load_model_configs(model_config_dir: Union[str, Path]) -> List[ModelConfig]:
    """Loads the model_config saved from ammo export.

    Args:
        model_config_dir: The directory where ammo exports the optimized model.
            Inside the directory, each gpu rank will have its own json and npz file.
            The json file represents the general ModelConfig structure while the detailed
            weights are stored in the npz file.
    Returns:
        The list of `ModelConfig` loaded and constructed.
    """
    model_config_dir = Path(model_config_dir)
    assert model_config_dir.is_dir()

    model_configs = []
    tensor_parallel = 0

    def _valid_json_filename(filename):
        pattern = r"^\w+_tp\d+_rank\d+\.json$"
        return bool(re.match(pattern, filename))

    for file_name in model_config_dir.iterdir():
        if file_name.suffix == ".json" and _valid_json_filename(file_name.name):
            with open(file_name, "r") as f:
                model_config = json.load(f)
                config_tensor_parallel = model_config["tensor_parallel"]
                config_rank = model_config["rank"]

                if tensor_parallel == 0:
                    tensor_parallel = config_tensor_parallel
                    model_configs = [{}] * tensor_parallel
                else:
                    assert tensor_parallel == config_tensor_parallel, "tensor_parallel not aligned between configs"

                model_configs[config_rank] = model_config

    for i, model_config in enumerate(model_configs):
        assert model_config, f"Failed to load model_config for rank {i}"
        decoder_type = model_config["layers"][0]["decoder_type"]
        weights_file = f"{decoder_type}_tp{tensor_parallel}_rank{i}.npz"
        weights = dict(np.load(model_config_dir / weights_file))
        _restore_model_config(model_config, weights)
        model_configs[i] = ModelConfig.from_dict(model_config)

    return model_configs
