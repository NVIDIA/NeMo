import torch
from tensorrt_llm.layers import MLP
from tensorrt_llm.models.gptj.model import GPTJAttention

from ..tensorrt_llm_utils import build_layernorm, torch_to_np
from .decoder import DecoderLayer


class GPTJDecoderLayer(DecoderLayer):
    def hidden_act_fn(self, layer):
        return layer.mlp.act

    def infer_hidden_size(self, layer):
        return layer.attn.embed_dim

    def infer_num_attention_heads(self, layer):
        return layer.attn.num_attention_heads

    def infer_max_position_embeddings(self, layer):
        return layer.attn.bias.shape[2]

    def build_input_layernorm(self, layer):
        return build_layernorm(layer.ln_1, dtype=self.dtype)

    def build_attention(self, layer):
        rotary_dim = layer.attn.rotary_dim
        attention = GPTJAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            rotary_dim=rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            dtype=self.dtype,
        )

        # Attention QKV Linear
        assert self.tensor_parallel == 1 and self.rank == 0, "Only single GPU is supported"

        # concatenate the Q, K, V layers weights.
        q_weights = layer.attn.q_proj.weight
        k_weights = layer.attn.k_proj.weight
        v_weights = layer.attn.v_proj.weight
        qkv_weights = torch.cat((q_weights, k_weights, v_weights))
        attention.qkv.weight.value = torch_to_np(qkv_weights, dtype=self.dtype)

        # Attention Dense (out_proj) Linear
        attention.dense.weight.value = torch_to_np(layer.attn.out_proj.weight, dtype=self.dtype)

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

        mlp.fc.weight.value = torch_to_np(layer.mlp.fc_in.weight, dtype=self.dtype)
        mlp.fc.bias.value = torch_to_np(layer.mlp.fc_in.bias, dtype=self.dtype)
        mlp.proj.weight.value = torch_to_np(layer.mlp.fc_out.weight, dtype=self.dtype)
        mlp.proj.bias.value = torch_to_np(layer.mlp.fc_out.bias, dtype=self.dtype)
        return mlp

    def build_post_layernorm(self, layer):
        # GPTJ do not have post layer_norm
        return None

    def post_attention_forward(self, residual, hidden_states, attention_output):
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attention_output + feed_forward_hidden_states + residual
        return hidden_states
