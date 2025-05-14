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

from nemo.export.huggingface.utils import ckpt_load, get_model, get_tokenizer, io_model_exporter, load_config
from nemo.lightning import io
from nemo.lightning.io.state import TransformFns, _ModelState
from nemo.utils import logging

MixtralModel = get_model("MixtralModel")


@io_model_exporter(MixtralModel, "hf")
class HFMixtralExporter(io.ModelConnector["MixtralModel", "MixtralForCausalLM"]):
    """NeMo to HF exporter"""

    def init(self) -> "MixtralForCausalLM":
        """HFMixtralExporter initialization"""
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        """export to hf format"""
        # TODO: Make it work with lazy init
        # with torch.device("meta"):
        #     target = self.init()
        source, source_config = ckpt_load(str(self))
        source = _ModelState(source, source_config)

        target = self.init()
        target = self.convert_state(source, target)

        # TODO: Make sure we don't need to do this
        target = target.cpu()
        target.save_pretrained(output_path)
        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            # TODO: Optional -- maybe fetch from remote HF?
            logging.warning(
                f"Only huggingface tokenizer is supported for Mixtral HF export. Tokenizer will not be saved."
            )

        return output_path

    def convert_state(self, source, target):
        """convert state"""

        # TODO: Make it work via state_transform
        experts = {k: v for k, v in source._state_dict.items() if "experts" in k}
        non_experts = {k: v for k, v in source._state_dict.items() if "experts" not in k}
        transformed_experts = {}
        for k, v in experts.items():
            for i in range(v.shape[0]):
                new_key = k.replace("experts.experts", f"experts.experts.{i}")
                transformed_experts[new_key] = v[i]

        source._state_dict = transformed_experts | non_experts

        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # MoE
            "decoder.layers.*.mlp.experts.experts.*.linear_fc2.weight": "model.layers.*.block_sparse_moe.experts.*.w2.weight",  # pylint: disable=line-too-long
            "decoder.layers.*.mlp.router.weight": "model.layers.*.block_sparse_moe.gate.weight",
            # lm-head
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
                source_key="decoder.layers.*.mlp.experts.experts.*.linear_fc1.weight",
                target_key=(
                    "model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    "model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                fn=TransformFns.split_fc1,
            ),
        ]

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return get_tokenizer(str(self))

    @property
    def config(self) -> "MixtralConfig":
        """return hf-config from mcore"""
        source = load_config(str(self))

        from transformers import MixtralConfig as HfMixtralConfig

        return HfMixtralConfig(
            architectures=["MixtralForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            max_position_embeddings=source.max_position_embeddings,
            seq_length=source.max_position_embeddings,
            # RoPe
            rope_theta=source.rotary_base,
            # transformer config
            num_attention_heads=source.num_attention_heads,
            num_key_value_heads=source.num_query_groups,
            num_local_experts=source.num_moe_experts,
            num_experts_per_tok=source.moe_router_topk,
            # norm
            rms_norm_eps=source.layernorm_epsilon,
            # init
            initializer_range=source.init_method_std,
            # vocab
            vocab_size=self.tokenizer.vocab_size,
            head_dim=source.kv_channels,
        )
