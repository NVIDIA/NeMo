import os
from dataclasses import dataclass
from typing import Union

import numpy as np
from tensorrt_llm.functional import is_gated_activation, non_gated_version
from tensorrt_llm.layers import MLP, Attention, AttentionMaskType, GatedMLP, LayerNorm
from transformers import GPT2Config

from ..tensorrt_llm_utils import get_tensor_from_file
from .decoder import DecoderLayer


@dataclass
class NemoLayer:
    weights_dir: Union[str, os.PathLike]
    i: int
    config: GPT2Config


class NemoDecoderLayer(DecoderLayer):
    def hidden_act_fn(self, layer):
        return layer.config.activation_function

    def infer_hidden_size(self, layer):
        return layer.config.n_embd

    def infer_num_attention_heads(self, layer):
        return layer.config.n_head

    def infer_max_position_embeddings(self, layer):
        return layer.config.n_positions

    def build_input_layernorm(self, layer):
        trt_layer = LayerNorm(normalized_shape=self.hidden_size, dtype=self.dtype)
        trt_layer.weight.value = get_tensor_from_file(
            layer.weights_dir, f"layers.{layer.i}.input_layernorm.weight", dtype=self.dtype
        )
        trt_layer.bias.value = get_tensor_from_file(
            layer.weights_dir, f"layers.{layer.i}.input_layernorm.bias", dtype=self.dtype
        )
        return trt_layer

    def build_attention(self, layer):
        rotary_pct = layer.config.rotary_pct
        position_embedding_type = "learned_absolute" if rotary_pct == 0.0 else "rope"

        attention = Attention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            apply_query_key_layer_scaling=False,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type=position_embedding_type,
            neox_rotary_style=True,
            rotary_embedding_percentage=rotary_pct,
            bias=layer.config.bias,
            multi_query_mode=False,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            use_int8_kv_cache=False,
        )

        n_embd = layer.config.n_embd
        c_attn_out_dim = 3 * n_embd // self.tensor_parallel
        attention.qkv.weight.value = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    layer.weights_dir,
                    f"layers.{layer.i}.attention.query_key_value.weight.{self.rank}",
                    shape=[n_embd, c_attn_out_dim],
                    dtype=self.dtype,
                ),
                [1, 0],
            )
        )

        if layer.config.bias:
            attention.qkv.bias.value = np.ascontiguousarray(
                get_tensor_from_file(
                    layer.weights_dir,
                    f"layers.{layer.i}.attention.query_key_value.bias.{self.rank}",
                    dtype=self.dtype,
                )
            )

        attention.dense.weight.value = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    layer.weights_dir,
                    f"layers.{layer.i}.attention.dense.weight.{self.rank}",
                    shape=[n_embd // self.tensor_parallel, n_embd],
                    dtype=self.dtype,
                ),
                [1, 0],
            )
        )

        if layer.config.bias:
            attention.dense.bias.value = get_tensor_from_file(
                layer.weights_dir,
                f"layers.{layer.i}.attention.dense.bias",
                dtype=self.dtype,
            )

        return attention

    def build_mlp(self, layer):
        gated = is_gated_activation(self.hidden_act)
        mlp_class = GatedMLP if gated else MLP

        n_embd = layer.config.n_embd
        inter_size = layer.config.intermediate_size

        mlp = mlp_class(
            self.hidden_size,
            inter_size,
            non_gated_version(self.hidden_act),
            layer.config.bias,
            self.dtype,
            self.tp_group,
            self.tensor_parallel,
        )

        mlp.fc.weight.value = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    layer.weights_dir,
                    f"layers.{layer.i}.mlp.dense_h_to_4h.weight.{self.rank}",
                    shape=[n_embd, inter_size // self.tensor_parallel],
                    dtype=self.dtype,
                ),
                [1, 0],
            )
        )

        if layer.config.bias:
            mlp.fc.bias.value = get_tensor_from_file(
                layer.weights_dir,
                f"layers.{layer.i}.mlp.dense_h_to_4h.bias.{self.rank}",
                dtype=self.dtype,
            )

        if gated:
            mlp.gate.weight.value = np.ascontiguousarray(
                np.transpose(
                    get_tensor_from_file(
                        layer.weights_dir,
                        f"layers.{layer.i}.mlp.dense_h_to_4h.gate.weight.{self.rank}",
                        shape=[n_embd, inter_size // self.tensor_parallel],
                        dtype=self.dtype,
                    ),
                    [1, 0],
                )
            )

        mlp.proj.weight.value = np.ascontiguousarray(
            np.transpose(
                get_tensor_from_file(
                    layer.weights_dir,
                    f"layers.{layer.i}.mlp.dense_4h_to_h.weight.{self.rank}",
                    shape=[inter_size // self.tensor_parallel, n_embd],
                    dtype=self.dtype,
                ),
                [1, 0],
            )
        )

        if layer.config.bias:
            mlp.proj.bias.value = get_tensor_from_file(
                layer.weights_dir, f"layers.{layer.i}.mlp.dense_4h_to_h.bias", dtype=self.dtype
            )

        return mlp

    def build_post_layernorm(self, layer):
        trt_layer = LayerNorm(normalized_shape=self.hidden_size, dtype=self.dtype)
        trt_layer.weight.value = get_tensor_from_file(
            layer.weights_dir, f"layers.{layer.i}.post_attention_layernorm.weight", dtype=self.dtype
        )
        trt_layer.bias.value = get_tensor_from_file(
            layer.weights_dir, f"layers.{layer.i}.post_attention_layernorm.bias", dtype=self.dtype
        )
        return trt_layer

    def post_attention_forward(self, residual, hidden_states, attention_output):
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
