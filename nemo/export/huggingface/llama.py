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

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nemo.export.huggingface.utils import (
    ckpt_load,
    get_model,
    get_tokenizer,
    io_model_exporter,
    load_config,
    torch_dtype_from_mcore_config,
)
from nemo.lightning import io
from nemo.lightning.io.state import TransformFns, _ModelState
from nemo.utils import logging

if TYPE_CHECKING:
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM


LlamaModel = get_model("LlamaModel")


# TODO: test multiple Llama model types, test Llama4
@io_model_exporter(LlamaModel, "hf")
class HFLlamaExporter(io.ModelConnector["LlamaModel", "LlamaForCausalLM"]):
    """Exporter for converting NeMo Llama models to Hugging Face format.

    This class handles the conversion of NeMo's LlamaModel to Hugging Face's
    LlamaForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self, dtype=torch.bfloat16) -> "LlamaForCausalLM":  # TODO precision
        """Initialize a HF LlamaForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            LlamaForCausalLM: Initialized HF Llama model
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from NeMo to HF format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved HF model
        """
        source, source_config = ckpt_load(str(self))
        if self.is_llama4(source_config):
            source = self._modify_llama4_source_state(source, source_config)
        else:
            source = _ModelState(source, source_config)

        target = self.init(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target, source_config)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(output_path, state_dict=state_dict)
        else:
            target.save_pretrained(output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target, source_config=None):
        # pylint: disable=C0301
        """Convert state dict from NeMo format to HF format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme.

        Args:
            source: Source NeMo model
            target: Target HF model
            source_config: Source NeMo config (optional, used for Llama4)

        Returns:
            The target model with weights transferred from source
        """

        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                fn=TransformFns.split_fc1,
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        if not self.config.tie_word_embeddings:
            transforms.append(
                io.state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                )
            )

        if self.is_llama4(source_config):
            # Llama4's Pre MLP LayerNorm is at decoder.layers.*.pre_mlp_layernorm.weight
            # instead of mlp.linear_fc1.layer_norm_weight
            mapping.pop("decoder.layers.*.mlp.linear_fc1.layer_norm_weight")
            mapping.update(
                {
                    "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
                    "decoder.layers.*.mlp.router.weight": "model.layers.*.feed_forward.router.weight",
                    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.feed_forward.shared_expert.down_proj.weight",
                    "decoder.layers.*.mlp.experts.linear_fc2.weight": "model.layers.*.feed_forward.experts.down_proj",
                    "decoder.layers.*.mlp.experts.linear_fc1.weight": "model.layers.*.feed_forward.experts.gate_up_proj",
                }
            )
            transforms.extend(
                [
                    io.state_transform(
                        source_key="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                        target_key=(
                            "model.layers.*.feed_forward.shared_expert.gate_proj.weight",
                            "model.layers.*.feed_forward.shared_expert.up_proj.weight",
                        ),
                        fn=TransformFns.split_fc1,
                    ),
                    io.state_transform(
                        source_key="decoder.layers.*.mlp.linear_fc1.weight",
                        target_key=(
                            "model.layers.*.feed_forward.gate_proj.weight",
                            "model.layers.*.feed_forward.up_proj.weight",
                        ),
                        fn=TransformFns.split_fc1,
                    ),
                ]
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self):
        """Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return get_tokenizer(str(self))

    @property
    def config(self) -> "HFLlamaConfig":
        """Create a HF LlamaConfig from the NeMo model config.

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFLlamaConfig: HF configuration for Llama models
        """
        source = load_config(str(self))

        if self.is_llama4(source):
            # Separate Llama4 from Llama2/3 for clarity
            return self.create_llama4_config(source)

        from transformers import LlamaConfig as HFLlamaConfig

        rope_scaling = None
        if hasattr(source, 'scale_factor') and hasattr(source, 'low_freq_factor'):
            rope_scaling = {
                'factor': source.scale_factor,
                'low_freq_factor': float(source.low_freq_factor),
                'high_freq_factor': float(source.high_freq_factor),
                'original_max_position_embeddings': source.old_context_len,
                'rope_type': 'llama3',
            }
        return HFLlamaConfig(
            architectures=["LlamaForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
        )

    @staticmethod
    def is_llama4(source_config):
        """Check if the model config is for Llama4."""
        return "llama4" in source_config._target_.lower()

    def create_llama4_config(self, source):
        """Create a HF Llama4TextConfig from the NeMo Llama4Config."""
        from transformers import Llama4TextConfig as HFLlama4TextConfig

        rope_scaling = (
            {
                'factor': source.rope_scaling_factor,
                'low_freq_factor': 1.0,
                'high_freq_factor': 4.0,
                'original_max_position_embeddings': 8192,
                'rope_type': 'llama3',
            }
            if source.rope_scaling
            else None
        )

        # MoE config
        moe_layer_freq = getattr(source, 'moe_layer_freq', None)
        interleave_moe_layer_step = None
        if moe_layer_freq is not None:
            if isinstance(moe_layer_freq, int):
                interleave_moe_layer_step = moe_layer_freq
            elif isinstance(moe_layer_freq, list):
                interleave_moe_layer_step = source.num_layers // sum(moe_layer_freq)
            else:
                raise ValueError(f'Unexpected moe_layer_freq {moe_layer_freq}')

        config = HFLlama4TextConfig(
            head_dim=source.kv_channels,
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.moe_ffn_hidden_size,
            intermediate_size_mlp=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            num_experts_per_tok=source.moe_router_topk,
            num_local_experts=source.num_moe_experts,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            use_qk_norm=source.qk_l2_norm,
            rope_theta=source.rotary_base,
            vocab_size=source.vocab_size,
            rope_scaling=rope_scaling,
            interleave_moe_layer_step=interleave_moe_layer_step,
            pad_token_id=self.tokenizer.tokens_to_ids(['<|finetune_right_pad|>'])[0],
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
            # no rope
        )
        return config

    def _modify_llama4_source_state(self, state_dict, source_config):
        """
        For MoE layer, we transpose the gate_up_proj and down_proj to match HF implementation.
        For dense layer, we change the name for the post attention layer norm to
        avoid the many-to-one mapping in the conversion confi.
        """

        for layer_i in range(source_config.num_layers):
            is_moe_layer = True
            if isinstance(source_config.moe_layer_freq, list):
                assert len(source_config.moe_layer_freq) == source_config.num_layers
                is_moe_layer = source_config.moe_layer_freq[layer_i]
            if is_moe_layer:
                # gate_up_proj
                weight = state_dict.pop(f"decoder.layers.{layer_i}.mlp.experts.experts.linear_fc1.weight")
                state_dict[f"decoder.layers.{layer_i}.mlp.experts.linear_fc1.weight"] = weight.permute(
                    0, 2, 1
                ).contiguous()

                # down_proj
                weight = state_dict.pop(f"decoder.layers.{layer_i}.mlp.experts.experts.linear_fc2.weight")
                state_dict[f"decoder.layers.{layer_i}.mlp.experts.linear_fc2.weight"] = weight.permute(
                    0, 2, 1
                ).contiguous()

            else:
                assert f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight" in state_dict
                weight = state_dict.pop(f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight")
                state_dict[f"decoder.layers.{layer_i}.pre_mlp_layernorm.weight"] = weight

        return _ModelState(state_dict, source_config)
