from typing import Optional

from tensorrt_llm.models.gptj.model import GPTJAttention

from ..model_config import LINEAR_COLUMN, LINEAR_ROW, AttentionConfig, LayernormConfig, LinearConfig, MLPConfig
from .decoder import DecoderLayer, DecoderLayerConfigBuilder


class GPTJDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    def hidden_act_fn(self, layer):
        return layer.mlp.act

    def infer_num_attention_heads(self, layer):
        return layer.attn.num_attention_heads

    def infer_max_position_embeddings(self, layer):
        return layer.attn.bias.shape[2]

    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.ln_1, dtype=self.dtype)

    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.attn.out_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.rotary_dim = layer.attn.rotary_dim

        return config

    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.fc_in,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.fc_out,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        # GPTJ do not have post layer_norm
        return None


class GPTJDecoderLayer(DecoderLayer):
    def build_attention(self, layer):
        assert self.tensor_parallel == 1 and self.rank == 0, "Only single GPU is supported for GPTJ"

        attention = GPTJAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            rotary_dim=layer.attention.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            dtype=self.dtype,
        )

        return attention

    def post_attention_forward(self, residual, hidden_states, attention_output):
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attention_output + feed_forward_hidden_states + residual
        return hidden_states
