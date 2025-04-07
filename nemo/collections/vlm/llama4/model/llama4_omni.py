# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from nemo.collections.llm import Llama4Config as Llama4TextConfig
from nemo.collections.llm import Llama4Experts16Config, Llama4Experts128Config
from nemo.collections.vlm import Llama4OmniConfig, Llama4OmniModel, Llama4VisionConfig, MultimodalProjectorConfig
from nemo.collections.vlm.neva.model.llava import import_qkv
from nemo.lightning import io, teardown
from nemo.lightning.io.state import TransformFns, _ModelState

try:
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


@dataclass
class Llama4ScoutExperts16Config(Llama4OmniConfig):
    """Llava v1.5 Config 7B"""

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama4Experts16Config())
    vision_transformer_config: TransformerConfig = field(default_factory=lambda: Llama4VisionConfig())
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mcore_affine",
            input_size=4096,
            hidden_size=5120,
            ffn_hidden_size=5120,
            bias=False,
            bias_activation_fusion=False,
        )
    )


@dataclass
class Llama4MaverickExperts128Config(Llama4OmniConfig):
    """Llava v1.5 Config 13B"""

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama4Experts128Config())
    vision_transformer_config: TransformerConfig = field(default_factory=lambda: Llama4VisionConfig())
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mcore_affine",
            input_size=4096,
            hidden_size=5120,
            ffn_hidden_size=5120,
            bias=False,
            bias_activation_fusion=False,
        )
    )


@io.model_importer(Llama4OmniModel, "hf")
class HFLlama4OmniImporter(io.ModelConnector["Llama4ForConditionalGeneration", Llama4OmniModel]):
    """Importer for converting Hugging Face Llama models to NeMo format.

    This class handles the conversion of Hugging Face's LlamaForCausalLM models
    to NeMo's Llama4OmniModel format, including weight mapping and configuration translation.
    """

    def init(self) -> Llama4OmniModel:
        """Initialize a NeMo Llama4OmniModel instance.

        Returns:
            Llama4OmniModel: Initialized NeMo Llama model with the appropriate configuration
                        and tokenizer.
        """
        return Llama4OmniModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """

        from transformers import Llama4ForConditionalGeneration

        source = Llama4ForConditionalGeneration.from_pretrained(str(self), torch_dtype='auto')

        target = self.init()
        trainer = self.nemo_setup(target)
        source = source.to(torch.bfloat16)
        target = target.to(torch.bfloat16)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Llama model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Convert state dict from HF format to NeMo format.

        Maps the weights from the HF model to the NeMo model according to
        the appropriate mapping scheme.

        Args:
            source: Source HF model
            target: Target NeMo model

        Returns:
            The result of applying the transforms
        """

        source = self._modify_llama4_source_state(source)
        mapping = {
            "vision_model.positional_embedding_vlm": "vision_model.position_embeddings.weight",
            "vision_model.patch_embedding.linear.weight": "vision_model.conv1._linear.weight",
            "vision_model.layernorm_pre.weight": "vision_model.ln_pre.weight",
            "vision_model.layernorm_pre.bias": "vision_model.ln_pre.bias",
            "vision_model.layernorm_post.weight": "vision_model.ln_post.weight",
            "vision_model.layernorm_post.bias": "vision_model.ln_post.bias",
            "vision_model.vision_adapter.mlp.fc1.weight": "vision_model.adapter.mlp.encoder.linear_fc1.weight",
            'vision_model.vision_adapter.mlp.fc2.weight': "vision_model.adapter.mlp.encoder.linear_fc2.weight",
            "vision_model.model.layers.*.self_attn.o_proj.weight": "vision_model.decoder.layers.*.self_attention.linear_proj.weight",
            "vision_model.model.layers.*.self_attn.o_proj.bias": "vision_model.decoder.layers.*.self_attention.linear_proj.bias",
            "vision_model.model.layers.*.input_layernorm.weight": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "vision_model.model.layers.*.input_layernorm.bias": "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "vision_model.model.layers.*.mlp.fc1.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.weight",
            "vision_model.model.layers.*.mlp.fc1.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.bias",
            "vision_model.model.layers.*.mlp.fc2.weight": "vision_model.decoder.layers.*.mlp.linear_fc2.weight",
            "vision_model.model.layers.*.mlp.fc2.bias": "vision_model.decoder.layers.*.mlp.linear_fc2.bias",
            "vision_model.model.layers.*.post_attention_layernorm.weight": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "vision_model.model.layers.*.post_attention_layernorm.bias": "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "multi_modal_projector.linear_1.weight": "vision_projection.encoder.weight",
            "language_model.model.embed_tokens.weight": "language_model.embedding.word_embeddings.weight",
            "language_model.model.layers.*.self_attn.o_proj.weight": "language_model.decoder.layers.*.self_attention.linear_proj.weight",
            "language_model.model.layers.*.input_layernorm.weight": "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "language_model.model.norm.weight": "language_model.decoder.final_layernorm.weight",
            "language_model.lm_head.weight": "language_model.output_layer.weight",
            # Post Attention LayerNorm
            "language_model.model.layers.*.post_attention_layernorm.weight": "language_model.decoder.layers.*.pre_mlp_layernorm.weight",
            "language_model.model.layers.*.dense-post_attention_layernorm.weight": "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            # MoE Router
            "language_model.model.layers.*.feed_forward.router.weight": "language_model.decoder.layers.*.mlp.router.weight",
            # MoE Shared Experts
            "language_model.model.layers.*.feed_forward.shared_expert.down_proj.weight": "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
            # MoE Experts
            "language_model.model.layers.*.feed_forward.experts.*.down_proj": "language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
            "language_model.model.layers.*.feed_forward.experts.*.gate_up_proj": "language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
            # Dense MLP (for moe_layer_freq != 1)
            "language_model.model.layers.*.feed_forward.down_proj.weight": "language_model.decoder.layers.*.mlp.linear_fc2.weight",
        }

        transforms = [
            _import_cls_token,
            _import_language_qkv,
            _import_vision_qkv,
            _import_vision_qkv_bias,
            io.state_transform(
                source_key=(
                    "language_model.model.layers.*.feed_forward.shared_expert.gate_proj.weight",
                    "language_model.model.layers.*.feed_forward.shared_expert.up_proj.weight",
                ),
                target_key="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
            io.state_transform(
                source_key=(
                    "language_model.model.layers.*.feed_forward.gate_proj.weight",
                    "language_model.model.layers.*.feed_forward.up_proj.weight",
                ),
                target_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]

        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Get the tokenizer for the HF model.

        Returns:
            AutoTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    def _modify_llama4_source_state(self, source: nn.Module) -> _ModelState:
        """
        In Llama4, HF weight for local experts are mapped with a single tensor.
        Pre-chunk it before convert_state.
        For dense layer, we change the name for the post attention layer norm to
        avoid the many-to-one mapping in the conversion.
        """
        state_dict = source.state_dict()
        num_experts = source.config.text_config.num_local_experts
        language_transformer_config = self.config.language_transformer_config
        for layer_i in range(language_transformer_config.num_layers):
            is_moe_layer = True
            if isinstance(language_transformer_config.moe_layer_freq, list):
                assert len(language_transformer_config.moe_layer_freq) == language_transformer_config.num_layers
                is_moe_layer = language_transformer_config.moe_layer_freq[layer_i]
            if is_moe_layer:
                # gate_up_proj
                weight = state_dict.pop(f"language_model.model.layers.{layer_i}.feed_forward.experts.gate_up_proj")
                weights = torch.chunk(weight, num_experts, dim=0)
                for expert_i, expert_weight in enumerate(weights):
                    state_dict[
                        f"language_model.model.layers.{layer_i}.feed_forward.experts.{expert_i}.gate_up_proj"
                    ] = expert_weight.squeeze().transpose(0, 1)
                # down_proj
                weight = state_dict.pop(f"language_model.model.layers.{layer_i}.feed_forward.experts.down_proj")
                weights = torch.chunk(weight, num_experts, dim=0)
                for expert_i, expert_weight in enumerate(weights):
                    state_dict[f"language_model.model.layers.{layer_i}.feed_forward.experts.{expert_i}.down_proj"] = (
                        expert_weight.squeeze().transpose(0, 1)
                    )
            else:
                weight = state_dict.pop(f"language_model.model.layers.{layer_i}.post_attention_layernorm.weight")
                state_dict[f"language_model.model.layers.{layer_i}.dense-post_attention_layernorm.weight"] = weight

        source = _ModelState(state_dict)
        return source

    @property
    def config(self) -> Llama4OmniConfig:
        """Create a NeMo Llama4OmniConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            Llama4OmniConfig: NeMo configuration for Llama models
        """
        from transformers import AutoConfig, GenerationConfig

        source = AutoConfig.from_pretrained(str(self))
        try:
            generation_config = GenerationConfig.from_pretrained(str(self))
        except Exception:
            generation_config = None

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        src_text_config = source.text_config
        args = {
            'moe_router_topk': src_text_config.num_experts_per_tok,
            'num_moe_experts': src_text_config.num_local_experts,
            'qk_l2_norm': src_text_config.use_qk_norm,
            'moe_shared_expert_intermediate_size': src_text_config.intermediate_size,
            'moe_ffn_hidden_size': src_text_config.intermediate_size,
        }
        if (
            getattr(src_text_config, 'rope_scaling', None) is not None
            and src_text_config.rope_scaling.get('rope_type') == 'llama3'
        ):
            args.update({'rope_scaling': True, 'rope_scaling_factor': src_text_config.rope_scaling.get("factor", 8.0)})
        else:
            args.update({'rope_scaling': False})
        if getattr(src_text_config, 'interleave_moe_layer_step', 1) != 1:
            assert src_text_config.num_hidden_layers % src_text_config.interleave_moe_layer_step == 0
            pattern = [0] * (src_text_config.interleave_moe_layer_step - 1) + [1]
            num_patterns = src_text_config.num_hidden_layers // src_text_config.interleave_moe_layer_step
            args.update({'moe_layer_freq': pattern * num_patterns})

        language_transformer_config = Llama4TextConfig(
            num_layers=src_text_config.num_hidden_layers,
            hidden_size=src_text_config.hidden_size,
            ffn_hidden_size=(
                src_text_config.intermediate_size
                if not getattr(src_text_config, 'intermediate_size_mlp', None)
                else src_text_config.intermediate_size_mlp
            ),
            num_attention_heads=src_text_config.num_attention_heads,
            init_method_std=src_text_config.initializer_range,
            layernorm_epsilon=src_text_config.rms_norm_eps,
            num_query_groups=src_text_config.num_key_value_heads,
            seq_length=src_text_config.max_position_embeddings,
            rotary_base=src_text_config.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(src_text_config.vocab_size),
            share_embeddings_and_output_weights=getattr(src_text_config, "tie_word_embeddings", False),
            vocab_size=src_text_config.vocab_size,
            kv_channels=getattr(src_text_config, "head_dim"),
            generation_config=generation_config,
            **args,
        )

        # vision config doesn't change
        vision_transformer_config = Llama4VisionConfig()

        vision_projection_config = MultimodalProjectorConfig(
            projector_type="mcore_affine",
            input_size=vision_transformer_config.output_dim,
            hidden_size=language_transformer_config.hidden_size,
            ffn_hidden_size=language_transformer_config.hidden_size,
            bias=False,
            bias_activation_fusion=False,
        )

        return Llama4OmniConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
            bf16=True,
            params_dtype=torch.bfloat16,
        )


@io.state_transform(
    source_key=("vision_model.class_embedding",),
    target_key="vision_model.class_token",
)
def _import_cls_token(ctx: io.TransformCTX, cls_token):
    # pylint: disable=C0115,C0116
    return cls_token.reshape(1, 1, -1)


@io.state_transform(
    source_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_language_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.language_transformer_config
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key=(
        "vision_model.model.layers.*.self_attn.q_proj.weight",
        "vision_model.model.layers.*.self_attn.k_proj.weight",
        "vision_model.model.layers.*.self_attn.v_proj.weight",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_vision_qkv(ctx: io.TransformCTX, q, k, v):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config
    return import_qkv(
        q,
        k,
        v,
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=megatron_config.hidden_size,
        head_size=megatron_config.kv_channels,
    )


@io.state_transform(
    source_key=(
        "vision_model.model.layers.*.self_attn.q_proj.bias",
        "vision_model.model.layers.*.self_attn.k_proj.bias",
        "vision_model.model.layers.*.self_attn.v_proj.bias",
    ),
    target_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_vision_qkv_bias(ctx: io.TransformCTX, q_bias, k_bias, v_bias):
    # pylint: disable=C0115,C0116
    megatron_config = ctx.target.config.vision_transformer_config
    return import_qkv(
        q_bias.unsqueeze(-1),
        k_bias.unsqueeze(-1),
        v_bias.unsqueeze(-1),
        head_num=megatron_config.num_attention_heads,
        num_query_groups=megatron_config.num_query_groups,
        heads_per_group=megatron_config.num_attention_heads // megatron_config.num_query_groups,
        hidden_size=1,
        head_size=megatron_config.kv_channels,
    ).squeeze(-1)
