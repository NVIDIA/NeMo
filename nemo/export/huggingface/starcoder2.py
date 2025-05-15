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

from nemo.export.huggingface.utils import ckpt_load, get_model, get_tokenizer, io_model_exporter, load_config
from nemo.lightning import io
from nemo.lightning.io.state import TransformFns, _ModelState

if TYPE_CHECKING:
    from transformers import Starcoder2ForCausalLM
    from transformers import Starcoder2Config as HFStarcoder2Config

Starcoder2Model = get_model("Starcoder2Model")


@io_model_exporter(Starcoder2Model, "hf", register=False)
class HFStarcoder2Exporter(io.ModelConnector["Starcoder2Model", "Starcoder2ForCausalLM"]):
    """
    Exporter for converting NeMo Starcoder2Model to Hugging Face format.

    This class handles the conversion of NeMo's Starcoder2Model to Hugging Face's
    Starcoder2ForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self) -> "Starcoder2ForCausalLM":
        """
        Initialize a HF Starcoder2ForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            Starcoder2ForCausalLM: Initialized HF Starcoder2 model
        """
        from transformers import Starcoder2ForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return Starcoder2ForCausalLM._from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        source, source_config = ckpt_load(str(self))
        source = _ModelState(source, source_config)

        target = self.init()
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """
        Convert state dict from NeMo format to HF format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme.

        Args:
            source: Source NeMo model
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.bias": "model.layers.*.self_attn.o_proj.bias",
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp.c_fc.weight",
            "decoder.layers.*.mlp.linear_fc1.bias": "model.layers.*.mlp.c_fc.bias",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.c_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.bias": "model.layers.*.mlp.c_proj.bias",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.layers.*.input_layernorm.bias",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.layers.*.post_attention_layernorm.bias",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.final_layernorm.bias": "model.norm.bias",
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
                source_key="decoder.layers.*.self_attention.linear_qkv.bias",
                target_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                fn=TransformFns.split_qkv_bias,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
            io.state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
        ]

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self):
        """
        Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return get_tokenizer(str(self))

    @property
    def config(self) -> "HFStarcoder2Config":
        """Create a HF HFStarcoder2Config from the NeMo model config.

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFStarcoder2Config: HF configuration for Starcoder2 models
        """
        source = load_config(str(self))

        from transformers import Starcoder2Config as HFStarcoder2Config

        return HFStarcoder2Config(
            architectures=["Starcoder2ForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            partial_rotary_factor=source.rotary_percent,
            vocab_size=self.tokenizer.vocab_size,
        )
