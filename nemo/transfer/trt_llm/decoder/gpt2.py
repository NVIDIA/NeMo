import numpy as np
from tensorrt_llm.layers import MLP, Attention, AttentionMaskType

from ..tensorrt_llm_utils import build_layernorm, split, torch_to_np
from .decoder import DecoderLayer


class GPT2DecoderLayer(DecoderLayer):
    def hidden_act_fn(self, layer):
        return layer.mlp.act

    def infer_hidden_size(self, layer):
        return layer.attn.embed_dim

    def infer_num_attention_heads(self, layer):
        return layer.attn.num_heads

    def infer_max_position_embeddings(self, layer):
        return layer.attn.bias.shape[2]

    def build_input_layernorm(self, layer):
        return build_layernorm(layer.ln_1, dtype=self.dtype)

    def build_attention(self, layer):
        attention = Attention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            apply_query_key_layer_scaling=False,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type="learned_absolute",
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
        )

        qkv_shape = layer.attn.c_attn.weight.shape
        # Decode the concat QKV weights and split them to different GPU rank.
        attention.qkv.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.attn.c_attn.weight, dtype=self.dtype).reshape(qkv_shape[0], 3, qkv_shape[-1] // 3),
                self.tensor_parallel,
                self.rank,
                dim=-1,
            )
            .reshape(qkv_shape[0], -1)
            .transpose()
        )
        attention.qkv.bias.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.attn.c_attn.bias, dtype=self.dtype).reshape(3, qkv_shape[-1] // 3),
                self.tensor_parallel,
                self.rank,
                dim=-1,
            ).reshape(-1)
        )

        attention.dense.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.attn.c_proj.weight, dtype=self.dtype).transpose(),
                self.tensor_parallel,
                self.rank,
                dim=1,
            )
        )
        attention.dense.bias.value = torch_to_np(layer.attn.c_proj.bias, dtype=self.dtype)

        return attention

    def build_mlp(self, layer):
        mlp = MLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.hidden_size * 4,
            hidden_act=self.hidden_act,
            dtype=self.dtype,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
        )

        mlp.fc.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.mlp.c_fc.weight, dtype=self.dtype).transpose(),
                self.tensor_parallel,
                self.rank,
            )
        )
        mlp.fc.bias.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.mlp.c_fc.bias, dtype=self.dtype),
                self.tensor_parallel,
                self.rank,
            )
        )

        mlp.proj.weight.value = np.ascontiguousarray(
            split(
                torch_to_np(layer.mlp.c_proj.weight, dtype=self.dtype).transpose(),
                self.tensor_parallel,
                self.rank,
                dim=1,
            )
        )
        mlp.proj.bias.value = torch_to_np(layer.mlp.c_proj.bias, dtype=self.dtype)

        return mlp

    def build_post_layernorm(self, layer):
        return build_layernorm(layer.ln_2, dtype=self.dtype)

    def post_attention_forward(self, residual, hidden_states, attention_output):
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
