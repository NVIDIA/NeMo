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
    from transformers import GemmaForCausalLM

GemmaModel = get_model("GemmaModel")


@io_model_exporter(GemmaModel, "hf", register=False)
class HFGemmaExporter(io.ModelConnector["GemmaModel", "GemmaForCausalLM"]):
    """ """

    def init(self) -> "GemmaForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        """ """
        target = self.init()
        source, source_config = ckpt_load(str(self))
        source = _ModelState(source, source_config)
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """ """
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
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
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
        ]

        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self):
        """ """
        return get_tokenizer(str(self))

    @property
    def config(self) -> "GemmaConfig":
        """ """
        source = load_config(str(self))

        from transformers import GemmaConfig as HFGemmaConfig

        return HFGemmaConfig(
            architectures=["GemmaForCausalLM"],
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
            vocab_size=self.tokenizer.vocab_size,
        )
