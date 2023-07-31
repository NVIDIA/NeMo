import numpy as np
import torch
from tensorrt_llm.layers import Attention, AttentionMaskType, GatedMLP

from ..tensorrt_llm_utils import build_layernorm, split, torch_to_np
from .decoder import DecoderLayer


class LLAMADecoderLayer(DecoderLayer):
    def hidden_act_fn(self, layer):
        return layer.mlp.act_fn

    def infer_hidden_size(self, layer):
        return layer.self_attn.hidden_size

    def infer_num_attention_heads(self, layer):
        return layer.self_attn.num_heads

    def infer_max_position_embeddings(self, layer):
        return layer.self_attn.max_position_embeddings

    def build_input_layernorm(self, layer):
        return build_layernorm(layer.input_layernorm, dtype=self.dtype)

    def build_attention(self, layer):
        attention = Attention(
            self.hidden_size,
            self.num_attention_heads,
            self.max_position_embeddings,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type="rope",
            neox_rotary_style=True,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
        )

        q_weight = layer.self_attn.q_proj.weight
        k_weight = layer.self_attn.k_proj.weight
        v_weight = layer.self_attn.v_proj.weight

        qkv_weight = torch_to_np(torch.stack([q_weight, k_weight, v_weight]), dtype=self.dtype)

        q_emb = qkv_weight.shape[1]
        model_emb = qkv_weight.shape[2]
        split_v = split(qkv_weight, self.tensor_parallel, self.rank, dim=1)
        split_v = split_v.reshape(3 * (q_emb // self.tensor_parallel), model_emb)
        attention.qkv.weight.value = np.ascontiguousarray(split_v)

        attention.dense.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.self_attn.o_proj.weight, dtype=self.dtype),
                self.tensor_parallel,
                self.rank,
                dim=1,
            )
        )

        return attention

    def build_mlp(self, layer):
        mlp_hidden_size = layer.mlp.gate_proj.out_features
        mlp = GatedMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            hidden_act=self.hidden_act,
            dtype=self.dtype,
            bias=False,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
        )

        mlp.gate.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.mlp.up_proj.weight, dtype=self.dtype),
                self.tensor_parallel,
                self.rank,
                dim=0,
            )
        )
        mlp.proj.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.mlp.down_proj.weight, dtype=self.dtype),
                self.tensor_parallel,
                self.rank,
                dim=1,
            )
        )
        mlp.fc.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.mlp.gate_proj.weight, dtype=self.dtype),
                self.tensor_parallel,
                self.rank,
                dim=0,
            )
        )

        return mlp

    def build_post_layernorm(self, layer):
        return build_layernorm(layer.post_attention_layernorm, dtype=self.dtype)

    def post_attention_forward(self, residual, hidden_states, attention_output):
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states
