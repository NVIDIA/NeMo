from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed
from megatron.core import parallel_state as ps
from megatron.core.inference_params import InferenceParams
from megatron.core.optimizer import OptimizerConfig
from transformers import LlavaNextForConditionalGeneration

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import LlamaConfig
from nemo.collections.vlm import Llava15Config7B, Llava15Config13B, NevaModel
from nemo.collections.vlm.llavanext.model.base import LLavanextConfig, MCoreLlavanextModel
from nemo.collections.vlm.neva.model.base import HFCLIPVisionConfig, MultimodalProjectorConfig
from nemo.collections.vlm.neva.model.llava import HFLlavaImporter
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule


@dataclass
class Llava16Config7B(Llava15Config7B):
    pass


@dataclass
class Llava16Config13B(Llava15Config13B):
    pass


class LLavanextModel(NevaModel):
    def __init__(
        self,
        config: LLavanextConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            optim=optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True)),
            tokenizer=tokenizer,
            model_transform=model_transform,
        )

    def configure_model(self) -> MCoreLlavanextModel:
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
    ) -> torch.Tensor:
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
        )

        return output_tensor


@io.model_importer(LLavanextModel, "hf")
class HFLlavaNextImporter(
    HFLlavaImporter,
    io.ModelConnector["LlavaNextForConditionalGeneration", LLavanextModel],
):
    def init(self) -> LLavanextModel:
        return LLavanextModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:

        source = LlavaNextForConditionalGeneration.from_pretrained(str(self))
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target, image_newline=True)
        print(f"Converted Llava next model to Nemo, saving to {output_path}")

        self.nemo_save(output_path, trainer)

        print(f"Converted Llava next model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    @property
    def config(self) -> LLavanextConfig:
        from transformers import LlavaConfig as HFLlavaConfig

        source = HFLlavaConfig.from_pretrained(str(self))
        text_conifg = source.text_config

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        language_transformer_config = LlamaConfig(
            num_layers=text_conifg.num_hidden_layers,
            hidden_size=text_conifg.hidden_size,
            ffn_hidden_size=text_conifg.intermediate_size,
            num_attention_heads=text_conifg.num_attention_heads,
            init_method_std=text_conifg.initializer_range,
            layernorm_epsilon=text_conifg.rms_norm_eps,
            num_query_groups=text_conifg.num_key_value_heads,
            rotary_base=text_conifg.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(text_conifg.vocab_size),
            share_embeddings_and_output_weights=False,
        )
        vision_transformer_config = HFCLIPVisionConfig(
            pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
        )
        vision_projection_config = MultimodalProjectorConfig(input_size=1024, hidden_size=4096, ffn_hidden_size=4096)

        output = LLavanextConfig(
            language_transformer_config=language_transformer_config,
            vision_transformer_config=vision_transformer_config,
            vision_projection_config=vision_projection_config,
            vision_feature_layer=source.vision_feature_layer,
        )

        return output


__all__ = [
    "LLavanextModel",
]
