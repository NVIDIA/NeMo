from typing import Optional

from tensorrt_llm.layers import Attention, AttentionMaskType, PositionEmbeddingType

from ..model_config import (
    LINEAR_COLUMN,
    LINEAR_ROW,
    AttentionConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
)
from .decoder import DecoderLayer, DecoderLayerConfigBuilder


class GPT2DecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    def hidden_act_fn(self, layer):
        return layer.mlp.act

    def infer_num_attention_heads(self, layer):
        return layer.attn.num_heads

    def infer_max_position_embeddings(self, layer):
        return layer.attn.bias.shape[2]

    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.ln_1, dtype=self.dtype)

    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.attn.c_attn],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.attn.c_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.c_fc,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.c_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.ln_2, dtype=self.dtype)


class GPT2DecoderLayer(DecoderLayer):
    def build_attention(self, layer):
        attention = Attention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            num_layers=self.num_layers,
            apply_query_key_layer_scaling=False,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type=PositionEmbeddingType.learned_absolute,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
        )

        return attention

    def post_attention_forward(self, residual, hidden_states, attention_output):
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
