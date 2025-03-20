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
# pylint: disable=line-too-long

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Union

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModel, AutoConfig

from nemo.collections.audio.parts.utils.audio import toeplitz
from nemo.collections.llm import Llama2Config7B, Llama2Config13B, LlamaConfig, Llama31Config8B
from nemo.collections.llm.utils import Config
from nemo.collections.vlm.neva.model.llava import LlavaConfig
from nemo.collections.vlm.neva.model.base import NevaConfig, NevaModel
from nemo.collections.vlm.vision.base import HFCLIPVisionConfig, MultimodalProjectorConfig
from nemo.collections.vlm.vision.radio import RADIO_25_h_Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.utils import logging

if TYPE_CHECKING:

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

@dataclass
class CosmosMegatronConfig(LlavaConfig):
    """Cosmos Megatron Base Config"""

    pixel_shuffle: bool = True

    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Llama31Config8B(make_vocab_size_divisible_by=512)
    )
    vision_transformer_config: TransformerConfig = field(
        default_factory=lambda: RADIO_25_h_Config(
            img_w=512, img_h=512, patch_dim=16,
        )
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            input_size=5120, hidden_size=4096, ffn_hidden_size=4096,
            normalization='LayerNorm', projector_type="mcore_mlp",
        )
    )



@dataclass
class CosmosMegatronRadioLlama8BConfig(CosmosMegatronConfig):
    """Cosmos Megatron 8B Config"""
    pass


class CosmosMegatronModel(NevaModel):
    """Cosmos Megatron Model NeMo Wrapper"""

    def __init__(
        self,
        config: Annotated[Optional[LlavaConfig], Config[LlavaConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or CosmosMegatronConfig(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)


class StateDictWrapper:
    def __init__(self, state_dict):
        """
        Wraps a dictionary in a PyTorch-compatible object.

        Args:
            state_dict (dict): Dictionary to wrap.
        """
        for key, value in state_dict.items():
            if "_extra_state" not in key:
                state_dict[key] = value.float()
        self._state_dict = state_dict

    def state_dict(self):
        """
        Returns the wrapped state dictionary.
        """
        return self._state_dict


@io.model_importer(CosmosMegatronModel, "pyt")
class CosmosMegatronImporter(io.ModelConnector["CosmosMegatronModel", CosmosMegatronModel]):
    """Cosmos Megatron Importer"""

    def init(self) -> CosmosMegatronModel:
        # pylint: disable=C0115,C0116
        return CosmosMegatronModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        # pylint: disable=C0115,C0116
        source = torch.load(str(self), weights_only=False)
        source = StateDictWrapper(source["model"])

        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        print(f"Converted Llava model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted Llava model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target,):
        # pylint: disable=C0115,C0116
        mapping = {
            k: k
            for k in source.state_dict().keys()
            if "_extra_state" not in k
        }
        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
        )

    @property
    def tokenizer(self) -> "AutoTokenizer":
        # pylint: disable=C0115,C0116
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
        tokenizer = AutoTokenizer("meta-llama/Llama-3.1-8B-Instruct")
        new_special_tokens = {
            "additional_special_tokens": [
                "<image>", "<img>", "</img>",
                "<quad>", "</quad>",
                "<ref>", "</ref>",
                "<box>", "</box>"
            ]
        }
        tokenizer.tokenizer.add_special_tokens(new_special_tokens)
        return tokenizer

    @property
    def config(self) -> CosmosMegatronConfig:
        # pylint: disable=C0115,C0116
        output = CosmosMegatronRadioLlama8BConfig()

        return output


@io.model_exporter(CosmosMegatronModel, "hf")
class HFCosmosMegatronExporter(io.ModelConnector[CosmosMegatronModel, "PreTrainedModel"]):
    """
    Exporter class for converting NeMo Cosmos Megatron model to HuggingFace format.

    Inherits:
        io.ModelConnector: Connector interface to handle setup, save, and load using the Lightning framework.

    Methods:
        init: Initializes a new HuggingFace NVLM_D_Model model instance.
        apply: Converts the NeMo model to HuggingFace format and saves it.
        convert_state: Maps and transforms the state dictionary from NeMo to HuggingFace format.
        config: Generates and returns the HuggingFace LLaVA config for the model.
    """

    def init(self) -> "PreTrainedModel":
        """
        Initializes a HuggingFace NVLM_D_Model model.

        Args:
            dtype: The data type to use for the model (default: torch.bfloat16)

        Returns:
            NVLM_D_Model: A HuggingFace NVLM_D_Model model initialized with the configuration.
        """
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModel.from_config(self.config, trust_remote_code=True)

    def apply(self, output_path: Path) -> Path:
        """
        Converts the NeMo Cosmos Megatron to HuggingFace format and saves it to the specified path.

        Args:
            output_path (Path): The path where the converted HuggingFace model will be saved.

        Returns:
            Path: The output path where the HuggingFace model was saved.
        """
        source, _ = self.nemo_load(str(self))
        target = self.init()
        target = target.to(source.dtype)
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target):
        # pylint: disable=C0115,C0116,line-too-long
        """
        Maps and transforms the state dictionary from NeMo to HuggingFace format.

        Args:
            source: The source NeMo model.
            target: The target HuggingFace model.

        Returns:
            The target HuggingFace model with the converted state.
        """
        # Define the state mapping from NeMo to HuggingFace
        mapping = {
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
        }

        # Map vision projection components
        mapping.update(
            {
                "vision_projection.encoder.linear_fc1.weight": "mlp1.1.weight",
                "vision_projection.encoder.linear_fc1.bias": "mlp1.1.bias",
                "vision_projection.encoder.linear_fc2.weight": "mlp1.3.weight",
                "vision_projection.encoder.linear_fc2.bias": "mlp1.3.bias",
                "vision_projection.encoder.linear_fc1.layer_norm_weight": "mlp1.0.weight",
                "vision_projection.encoder.linear_fc1.layer_norm_bias": "mlp1.0.bias",
            }
        )

        # Map vision model components
        mapping.update(
            {
                "vision_model.position_embeddings": "vision_model.radio_model.model.patch_generator.pos_embed",
                "vision_model.embedder.weight": "vision_model.radio_model.model.patch_generator.embedder.weight",
                "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "vision_model.radio_model.model.blocks.*.norm1.weight",
                "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "vision_model.radio_model.model.blocks.*.norm1.bias",
                "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "vision_model.radio_model.model.blocks.*.norm2.weight",
                "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "vision_model.radio_model.model.blocks.*.norm2.bias",
                "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "vision_model.radio_model.model.blocks.*.attn.proj.weight",
                "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "vision_model.radio_model.model.blocks.*.attn.proj.bias",
                "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "vision_model.radio_model.model.blocks.*.mlp.fc1.weight",
                "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "vision_model.radio_model.model.blocks.*.mlp.fc1.bias",
                "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "vision_model.radio_model.model.blocks.*.mlp.fc2.weight",
                "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "vision_model.radio_model.model.blocks.*.mlp.fc2.bias",
            }
        )

        # Add transformations for specialized tensor manipulations
        transforms = [
            _export_language_qkv,
            _export_language_linear_fc1,
            _export_vision_qkv,
            _export_vision_qkv_bias,
            _export_class_token,
            _export_embedding,
            _export_language_head,
        ]

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """
        Gets the tokenizer from the loaded model context.

        Returns:
            The tokenizer specification.
        """
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self) -> "PretrainedConfig":
        """
        Generates the configuration for the HuggingFace NVLM_D_Model model based on the NeMo model.

        Returns:
            PretrainedConfig: A configuration object for the HuggingFace NVLM_D_Model model.
        """
        # from transformers import LlamaConfig as HFLlamaConfig

        # source = io.load_context(str(self), subpath="model.config")
        # language_config = source.language_transformer_config
        # vision_config = AutoConfig.from_pretrained("nvidia/C-RADIOv2-H", trust_remote_code=True)

        # # Create text config for HuggingFace model
        # text_config = HFLlamaConfig(
        #     num_hidden_layers=language_config.num_layers,
        #     hidden_size=language_config.hidden_size,
        #     intermediate_size=language_config.ffn_hidden_size,
        #     num_attention_heads=language_config.num_attention_heads,
        #     max_position_embeddings=language_config.seq_length,
        #     initializer_range=language_config.init_method_std,
        #     rms_norm_eps=language_config.layernorm_epsilon,
        #     num_key_value_heads=language_config.num_query_groups,
        #     rope_theta=language_config.rotary_base,
        #     vocab_size=self.tokenizer.vocab_size,
        #     tie_word_embeddings=language_config.share_embeddings_and_output_weights,
        # )
        return AutoConfig.from_pretrained("/ws/llama_3p1_8b_cradio_h_v2_hf", trust_remote_code=True)

# Define transformation functions needed for the exporter
@io.state_transform(
    source_key="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "language_model.model.layers.*.self_attn.q_proj.weight",
        "language_model.model.layers.*.self_attn.k_proj.weight",
        "language_model.model.layers.*.self_attn.v_proj.weight",
    ),
)
def _export_language_qkv(ctx: io.TransformCTX, linear_qkv):
    """Transforms NeMo's fused QKV weights to separate Q, K, V weights for HuggingFace format."""
    megatron_config = ctx.source.config.language_transformer_config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num // num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, hidden_size])
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


@io.state_transform(
    source_key="language_model.decoder.layers.*.mlp.linear_fc1.weight",
    target_key=(
        "language_model.model.layers.*.mlp.gate_proj.weight",
        "language_model.model.layers.*.mlp.up_proj.weight",
    ),
)
def _export_language_linear_fc1(ctx: io.TransformCTX, linear_fc1):
    """Splits NeMo's fused MLP linear_fc1 weight into gate_proj and up_proj for HuggingFace format."""
    gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)
    return gate_proj, up_proj


@io.state_transform(
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
    target_key="vision_model.radio_model.model.blocks.*.attn.qkv.weight",
)
def _export_vision_qkv(ctx: io.TransformCTX, linear_qkv):
    """Transforms NeMo's fused vision QKV weights to QKV weights for HuggingFace format."""
    megatron_config = ctx.source.config.vision_transformer_config

    hidden_size = megatron_config.hidden_size
    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups or head_num
    head_size = megatron_config.kv_channels or (hidden_size // head_num)

    order = torch.ones(3 * hidden_size).long()
    for j in range(num_query_groups):
        for i in range(head_size):
            order[j*head_size+i] = i + head_size*3*j
            order[j*head_size+i+num_query_groups*head_size] = head_size + i + head_size*3*j
            order[j*head_size+i+num_query_groups*head_size*2] = head_size*2 + i + head_size*3*j

    return linear_qkv[order]


@io.state_transform(
    source_key="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
    target_key="vision_model.radio_model.model.blocks.*.attn.qkv.bias",
)
def _export_vision_qkv_bias(ctx: io.TransformCTX, linear_qkv):
    """Transforms NeMo's fused vision QKV bias to QKV bias for HuggingFace format."""
    megatron_config = ctx.source.config.vision_transformer_config

    hidden_size = megatron_config.hidden_size
    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups or head_num
    head_size = megatron_config.kv_channels or (hidden_size // head_num)

    order = torch.ones(3 * hidden_size).long()
    for j in range(num_query_groups):
        for i in range(head_size):
            order[j*head_size+i] = i + head_size*3*j
            order[j*head_size+i+num_query_groups*head_size] = head_size + i + head_size*3*j
            order[j*head_size+i+num_query_groups*head_size*2] = head_size*2 + i + head_size*3*j

    return linear_qkv[order]


@io.state_transform(
    source_key="vision_model.class_token",
    target_key=(
        "vision_model.radio_model.model.patch_generator.cls_token.token",
        "radio_model.input_conditioner.norm_mean",
        "radio_model.input_conditioner.norm_std",
        "radio_model.summary_idxs",
    ),
)
def _export_class_token(ctx: io.TransformCTX, class_token):
    """Use class token transform to add constant weight for HuggingFace format."""
    norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(-1).unsqueeze(-1).cuda()
    norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).unsqueeze(-1).unsqueeze(-1).cuda()
    summary_idxs = torch.tensor([0, 1, 2]).cuda()
    return class_token, norm_mean, norm_std, summary_idxs


@io.state_transform(
    source_key="language_model.embedding.word_embeddings.weight",
    target_key="language_model.model.embed_tokens.weight",
)
def _export_embedding(ctx: io.TransformCTX, embedding):
    """Transforms the word embeddings from NeMo to HuggingFace format."""
    hf_config = ctx.target.config.llm_config
    # Prune any padding to match the HuggingFace vocab size
    return embedding[: hf_config.vocab_size, :]


@io.state_transform(
    source_key="language_model.output_layer.weight",
    target_key="language_model.lm_head.weight",
)
def _export_language_head(ctx: io.TransformCTX, output_weight):
    """Transforms the output layer from NeMo to HuggingFace format."""
    hf_config = ctx.target.config.llm_config
    # Prune any padding to match the HuggingFace vocab size
    return output_weight[: hf_config.vocab_size, :]
