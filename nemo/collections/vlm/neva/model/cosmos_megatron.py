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

from nemo.collections.audio.parts.utils.audio import toeplitz
from nemo.collections.llm import Llama2Config7B, Llama2Config13B, LlamaConfig, Llama31Config8B
from nemo.collections.llm.utils import Config
from nemo.collections.vlm import LlavaConfig
from nemo.collections.vlm.neva.model.base import NevaConfig, NevaModel
from nemo.collections.vlm.vision.base import HFCLIPVisionConfig, MultimodalProjectorConfig
from nemo.collections.vlm.vision.radio import RADIO_25_h_Config
from nemo.lightning import OptimizerModule, io, teardown

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
class CosmosMegatron8BConfig(CosmosMegatronConfig):
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
        output = CosmosMegatron8BConfig()

        return output
