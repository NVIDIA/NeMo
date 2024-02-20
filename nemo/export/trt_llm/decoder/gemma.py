from typing import Optional

from tensorrt_llm.layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, FusedGatedMLP, GatedMLP, MoeConfig,
                       PositionEmbeddingType, PromptTuningEmbedding, RmsNorm)

from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import AttentionMaskType, PositionEmbeddingType, MoeConfig
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.module import Module
from typing_extensions import override

from ..model_config import (
    LINEAR_COLUMN,
    LINEAR_ROW,
    AttentionConfig,
    LayernormConfig,
    LinearConfig,
    MLPConfig,
)
from .decoder import DecoderLayerBuilder, DecoderLayerConfigBuilder

class GemmaDecoderLayer(Module):

    def __init__(self,
        layer_id,
        hidden_size,
        num_attention_heads,
        num_kv_heads=None,
        head_size=None,
        max_position_embeddings=2048,
        dtype=None,
        attention_mask_type=AttentionMaskType.causal,
        hidden_act='silu',
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        rotary_base=10000.0,
        rotary_scaling=None,
        mlp_hidden_size=None,
        tp_group=None,
        tp_size=1,
        tp_rank=0,
        quant_mode=QuantMode(0),
        rms_norm_eps=1e-06,
        attn_bias=False,
        mlp_bias=False,
        use_fused_mlp=False,
        moe_config: MoeConfig = MoeConfig()):

        super().__init__()
        self._layer_id = layer_id  # useful for debugging
        # used for quantizing model
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        self.hidden_act = hidden_act
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.mlp_hidden_size = mlp_hidden_size
        self.attention_mask_type = attention_mask_type
        self.position_embedding_type = position_embedding_type
        self.input_layernorm = RmsNorm(normalized_shape=hidden_size,
                                    eps=rms_norm_eps,
                                    dtype=dtype)

        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            attention_head_size=head_size,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=rotary_base,
            rotary_embedding_scaling=rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=quant_mode,
            instance_id=2 * layer_id,
        )

        if not mlp_hidden_size:
            self.mlp_hidden_size = hidden_size * 4

        ClsMLP = GatedMLP
        mlp_kwargs = {}
        if moe_config.has_moe():
            ClsMLP = MOE
            mlp_kwargs = {
                "moe_config": moe_config,
                "tp_rank": tp_rank,
            }
        elif use_fused_mlp:
            ClsMLP = FusedGatedMLP
        self.mlp = ClsMLP(hidden_size=hidden_size,
                            ffn_hidden_size=self.mlp_hidden_size,
                            hidden_act=hidden_act,
                            dtype=dtype,
                            bias=mlp_bias,
                            tp_group=tp_group,
                            tp_size=tp_size,
                            quant_mode=quant_mode,
                            instance_id=2 * layer_id + 1,
                            **mlp_kwargs)
        self.post_layernorm = RmsNorm(normalized_shape=hidden_size,
                                        eps=rms_norm_eps,
                                        dtype=dtype)


    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                kv_cache_params=None,
                attention_params=None,
                all_reduce_workspace=None,
                lora_layer_params=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm0", hidden_states)

        attention_output = self.attention(hidden_states,
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          kv_cache_params=kv_cache_params,
                                          attention_params=attention_params,
                                          workspace=all_reduce_workspace,
                                          lora_layer_params=lora_layer_params)

        if use_cache:
            attention_output, presents = attention_output
        if self._layer_id == 0:
            self.register_network_output(f"attn", attention_output)

        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        if self._layer_id == 0:
            self.register_network_output(f"norm1", hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 all_reduce_workspace,
                                 lora_layer_params=lora_layer_params)
        if self._layer_id == 0:
            self.register_network_output(f"mlp", hidden_states)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states



class GemmaDecoderLayerConfigBuilder(DecoderLayerConfigBuilder):
    """The LLAMA implementation of the DecoderLayerConfigBuilder."""

    @override
    def hidden_act_fn(self, layer):
        return layer.mlp.act_fn

    @override
    def infer_num_attention_heads(self, layer):
        return layer.self_attn.num_heads

    @override
    def infer_num_kv_heads(self, layer):
        return layer.self_attn.num_key_value_heads

    @override
    def infer_max_position_embeddings(self, layer):
        return layer.self_attn.max_position_embeddings

    @override
    def build_input_layernorm(self, layer) -> LayernormConfig:
        return LayernormConfig.from_nn_module(layer.input_layernorm, dtype=self.dtype)

    @override
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

    @override
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

    @override
    def build_post_layernorm(self, layer) -> Optional[LayernormConfig]:
        return LayernormConfig.from_nn_module(layer.post_attention_layernorm, dtype=self.dtype)

class GemmaDecoderLayerBuilder(DecoderLayerBuilder):
    """The LLAMA implementation of the DecoderLayer."""

    @override
    def build_decoder(self, layer):
        rotary_scaling = None
        if layer.rotary_scaling is not None:
            rotary_scaling = {
                "type": "linear",
                "factor": float(layer.rotary_scaling)
            }

        moe_config = MoeConfig()
        if not layer.moe_num_experts is None:
            if layer.moe_top_k is None:
                layer.moe_top_k = 1

            layer.moe_tp_mode = MoeConfig.ParallelismMode.TENSOR_PARALLEL if layer.moe_tp_mode is None else None
            layer.moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE if layer.moe_renorm_mode is None else None
            moe_config = MoeConfig(layer.moe_num_experts, layer.moe_top_k,
                                   layer.moe_tp_mode, layer.moe_renorm_mode).validate()

        return GemmaDecoderLayer(
            layer_id=self.layer_id,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=layer.kv_channels,
            max_position_embeddings=self.max_position_embeddings,
            dtype=self.dtype,
            attention_mask_type=AttentionMaskType.causal,
            hidden_act=non_gated_version(self.hidden_act),
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            mlp_hidden_size=layer.ffn_hidden_size_local * self.tensor_parallel,
            tp_group=self.tp_group,
            tp_size=self.tensor_parallel,
            rotary_base=layer.rotary_base,
            rotary_scaling=rotary_scaling,
            moe_config=moe_config,
        )