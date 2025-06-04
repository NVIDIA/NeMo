# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Callable, List, Optional, Union

import torch

from megatron.core.inference_params import InferenceParams
from megatron.core.optimizer import OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.avlm.model.base import AVLMConfig
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import Llama3Config8B
from nemo.collections.speechlm.modules.asr_module import ASRModuleConfig
from nemo.collections.vlm.vision.base import HFCLIPVisionConfig, MultimodalProjectorConfig
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule


@dataclass
class AVLMConfig8B(AVLMConfig):
    """
    Configuration class for the 8B parameter variant of the AVLM model.

    """

    from transformers import PretrainedConfig

    language_transformer_config: TransformerConfig = field(default_factory=lambda: Llama3Config8B())
    vision_transformer_config: Union[TransformerConfig, PretrainedConfig] = field(
        default_factory=lambda: HFCLIPVisionConfig(
            pretrained_model_name_or_path="openai/clip-vit-large-patch14-336",
        )
    )
    vision_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mlp2x_gelu", input_size=1024, hidden_size=4096, ffn_hidden_size=4096
        )
    )
    audio_transformer_config: TransformerConfig = field(
        default_factory=lambda: ASRModuleConfig(
            _target_="nemo.collections.speechlm.modules.asr_module.ASRModuleConfig",
            use_hf_auto_model=True,
            hf_trust_remote_code=False,
            hf_load_pretrained_weights=True,
            pretrained_model="openai/whisper-large-v3",
            hidden_size=1280,
            target_module="model.encoder",
        )
    )
    audio_projection_config: TransformerConfig = field(
        default_factory=lambda: MultimodalProjectorConfig(
            projector_type="mlp2x_gelu", input_size=1280, hidden_size=4096, ffn_hidden_size=4096
        )
    )


class LlavaNextModel(NevaModel):
    """
    The LLaVA Next model class, extending NevaModel.

    Attributes:
        config (LlavaNextConfig): Configuration object for the model.
        optim (Optional[OptimizerModule]): Optimizer module. Defaults to a Megatron optimizer.
        tokenizer (Optional[TokenizerSpec]): Tokenizer specification for processing text inputs.
        model_transform (Optional[Callable[[torch.nn.Module], torch.nn.Module]]):
            Optional transformation applied to the model after initialization.
    """

    def __init__(
        self,
        config: LlavaNextConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
    ):
        """
        Initializes the LlavaNextModel.

        Args:
            config (LlavaNextConfig): Configuration object for the model.
            optim (Optional[OptimizerModule]): optimizer module. Defaults to Megatron optimizer.
            tokenizer (Optional[TokenizerSpec]): Optional tokenizer specification for processing text inputs.
            model_transform (Optional[Callable[[torch.nn.Module], torch.nn.Module]]):
                Optional transformation function applied to the model after initialization.
        """
        super().__init__(
            config=config,
            optim=optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True)),
            tokenizer=tokenizer,
            model_transform=model_transform,
        )

    def configure_model(self) -> MCoreLlavaNextModel:
        """
        Configures the underlying model instance if it has not been initialized.

        Returns:
            MCoreLlavaNextModel: The configured model instance.
        """
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        image_sizes: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        media: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
        num_media_tiles: Optional[List[int]] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        """
        Performs the forward pass of the LLaVA Next model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch, text_seq_len].
            position_ids (torch.Tensor): Position IDs of shape [batch, text_seq_len].
            image_sizes (torch.Tensor): Raw image sizes before tiling, of shape [batch, 2].
            loss_mask (Optional[torch.Tensor]): Text loss mask of shape [batch, text_seq_len].
            attention_mask (Optional[torch.Tensor]): Attention mask shape [batch, text_seq_len].
            media (Optional[torch.Tensor]): Input media tensor.
            labels (Optional[torch.Tensor]): Target labels of shape [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters.
            num_media_tiles (Optional[List[int]]): Number of tiles per image. Default assumes 1 tile per image.

        Returns:
            torch.Tensor: The model output. Shape depends on whether labels are provided.
            - If `labels` is provided: Loss tensor of shape [batch, seq_len].
            - If `labels` is not provided: Logits tensor of shape [batch, seq_len, vocab_size].
        """
        output_tensor = self.module(
            media=media,
            input_ids=input_ids,
            position_ids=position_ids,
            image_sizes=image_sizes,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
            num_media_tiles=num_media_tiles,
            packed_seq_params=packed_seq_params,
        )

        return output_tensor


__all__ = [
    "LlavaNextModel",
]
