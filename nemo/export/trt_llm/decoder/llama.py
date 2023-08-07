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


class LLAMADecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    def hidden_act_fn(self, layer):
        return layer.mlp.act_fn

    def infer_num_attention_heads(self, layer):
        return layer.self_attn.num_heads

    def infer_max_position_embeddings(self, layer):
        return layer.self_attn.max_position_embeddings

    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.input_layernorm, dtype=self.dtype)

    def build_attention(self, layer) -> AttentionConfig:
        config = AttentionConfig()
        config.qkv = LinearConfig.from_qkv_nn_modules(
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        config.dense = LinearConfig.from_nn_module(
            layer.self_attn.o_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    def build_mlp(self, layer) -> MLPConfig:
        config = MLPConfig()
        config.fc = LinearConfig.from_nn_module(
            layer.mlp.gate_proj,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.proj = LinearConfig.from_nn_module(
            layer.mlp.down_proj,
            LINEAR_ROW,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )
        config.gate = LinearConfig.from_nn_module(
            layer.mlp.up_proj,
            LINEAR_COLUMN,
            rank=self.rank,
            tensor_parallel=self.tensor_parallel,
            dtype=self.dtype,
        )

        return config

    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.post_attention_layernorm, dtype=self.dtype)


class LLAMADecoderLayer(DecoderLayer):
    def build_attention(self, layer):
        attention = Attention(
            self.hidden_size,
            self.num_attention_heads,
            self.max_position_embeddings,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=False,
            position_embedding_type=PositionEmbeddingType.rope,
            neox_rotary_style=True,
            multi_query_mode=False,
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
