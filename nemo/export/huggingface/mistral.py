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

import torch
import yaml

from nemo.export.huggingface.utils import ckpt_load, get_model, get_tokenizer, io_model_exporter
from nemo.lightning import io
from nemo.lightning.io.state import TransformFns, _ModelState
from nemo.utils import logging

MistralModel = get_model("MistralModel")


@io_model_exporter(MistralModel, "hf")
class HFMistralExporter(io.ModelConnector["MistralModel", "MistralForCausalLM"]):
    """ """

    def init(self, dtype=torch.bfloat16) -> "MistralForCausalLM":
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        # TODO: Make it work with lazy init
        # with torch.device("meta"):
        #     target = self.init()
        source, source_config = ckpt_load(str(self))
        source = _ModelState(source, source_config)

        target = self.init(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target)

        # TODO: Make sure we don't need to do this
        target = target.cpu()
        target.save_pretrained(output_path)
        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target):
        """ """
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
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
        transformed = io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )
        return transformed

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return get_tokenizer(str(self))

    @property
    def config(self) -> "MistralConfig":
        """ """
        model_yaml = Path(str(self)) / "context" / "model.yaml"
        if not model_yaml.exists():
            raise FileNotFoundError("model.yaml is not found in the context folder of the checkpoint.")
        with open(model_yaml, 'r') as stream:
            config = yaml.safe_load(stream)
        dict_to_obj = lambda d: (
            type('Config', (), {kk: dict_to_obj(vv) for kk, vv in d.items()}) if isinstance(d, dict) else d
        )

        if config['config']['seq_length'] is None:
            config['config']['seq_length'] = 32768
            logging.warning("seq_length is None, setting to 32768")

        source = dict_to_obj(config['config'])

        from transformers import MistralConfig as HfMistralConfig

        return HfMistralConfig(
            architectures=["MistralForCausalLM"],
            sliding_window=source.window_size[0] if source.window_size is not None else None,
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            head_dim=source.kv_channels,
        )
