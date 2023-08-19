from tensorrt_llm.layers import Attention, AttentionMaskType, PositionEmbeddingType

from .decoder import DecoderLayer


class GPTNextDecoderLayer(DecoderLayer):
    def build_attention(self, layer):
        rotary_pct = layer.rotary_pct
        position_embedding_type = (
            PositionEmbeddingType.learned_absolute if rotary_pct == 0.0 else PositionEmbeddingType.rope
        )
        bias_qkv = layer.attention.qkv.bias is not None
        bias_dense = layer.attention.dense.bias is not None
        assert bias_qkv == bias_dense

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
            bias=bias_qkv,
            multi_query_mode=False,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            use_int8_kv_cache=False,
        )

        return attention

    def post_attention_forward(self, residual, hidden_states, attention_output):
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
