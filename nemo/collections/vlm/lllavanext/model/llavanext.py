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
    """
    Configuration class for the 7B parameter variant of the LLaVA 16 model.

    Inherits all attributes and methods from Llava15Config7B without modification.
    """

    pass


@dataclass
class Llava16Config13B(Llava15Config13B):
    """
    Configuration class for the 13B parameter variant of the LLaVA 16 model.

    Inherits all attributes and methods from Llava15Config13B without modification.
    """

    pass


class LLavanextModel(NevaModel):
    """
    The LLaVA Next model class, extending NevaModel.

    Attributes:
        config (LLavanextConfig): Configuration object for the model.
        optim (Optional[OptimizerModule]): Optimizer module for training the model. Defaults to a Megatron optimizer.
        tokenizer (Optional[TokenizerSpec]): Tokenizer specification for processing text inputs.
        model_transform (Optional[Callable[[torch.nn.Module], torch.nn.Module]]):
            Optional transformation applied to the model after initialization.
    """

    def __init__(
        self,
        config: LLavanextConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
    ):
        """
        Initializes the LLavanextModel.

        Args:
            config (LLavanextConfig): Configuration object for the model.
            optim (Optional[OptimizerModule]): Optional optimizer module. If not provided, a default Megatron optimizer is used.
            tokenizer (Optional[TokenizerSpec]): Optional tokenizer specification for processing text inputs.
            model_transform (Optional[Callable[[torch.nn.Module], torch.nn.Module]]):
                Optional transformation function applied to the model after initialization.
        """
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            optim=optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True)),
            tokenizer=tokenizer,
            model_transform=model_transform,
        )

    def configure_model(self) -> MCoreLlavanextModel:
        """
        Configures the underlying model instance if it has not been initialized.

        Returns:
            MCoreLlavanextModel: The configured model instance.
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
    ) -> torch.Tensor:
        """
        Performs the forward pass of the LLaVA Next model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch, text_seq_len].
            position_ids (torch.Tensor): Position IDs of shape [batch, text_seq_len].
            image_sizes (torch.Tensor): Raw image sizes before tiling, of shape [batch, 2].
            loss_mask (Optional[torch.Tensor]): Text loss mask of shape [batch, text_seq_len].
            attention_mask (Optional[torch.Tensor]): Attention mask (before merging image embeddings) of shape [batch, text_seq_len].
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
        )

        return output_tensor


@io.model_importer(LLavanextModel, "hf")
class HFLlavaNextImporter(
    HFLlavaImporter,
    io.ModelConnector["LlavaNextForConditionalGeneration", LLavanextModel],
):
    """
    Importer class for converting HuggingFace LLaVA Next checkpoint to NeMo format.

    Inherits:
        HFLlavaImporter: Base class for HuggingFace LLaVA model importers.
        io.ModelConnector: Connector interface to handle setup, save, and load using the Lightning framework.

    Methods:
        init: Initializes a new LLavanextModel instance.
        apply: Converts the HuggingFace model to NeMo format and saves it.
        config: Generates and returns the LLavanextConfig for the model.
    """

    def init(self) -> LLavanextModel:
        """
        Initializes the LLavanextModel.

        Returns:
            LLavanextModel: An instance of the LLaVA Next model initialized with the configuration.
        """
        return LLavanextModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """
        Converts the HuggingFace LLaVA Next model to NeMo format and saves it to the specified path.

        Args:
            output_path (Path): The path where the converted NeMo model will be saved.

        Returns:
            Path: The output path where the NeMo model was saved.
        """

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
        """
        Generates the configuration for the LLaVA Next model based on the HuggingFace model.

        Returns:
            LLavanextConfig: A configuration object for the LLaVA Next model.
        """
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