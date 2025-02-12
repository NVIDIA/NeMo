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

from typing import Callable, Dict, List, Optional

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from megatron.core.inference_params import InferenceParams
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm import fn
from nemo.collections.multimodal.mimo.model.config import MimoConfig
from nemo.collections.multimodal.mimo.model.loss import MimoLossReduction
from nemo.collections.vlm import NevaModel
from nemo.lightning import OptimizerModule, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule


class MimoModel(NevaModel, L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: MimoConfig,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None
        self.wandb_logger = None

    def on_fit_start(self):
        # setup wandb logger to MimoLossReduction
        if self.trainer and self.config.stage in ['decoder_alignment']:
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    self.wandb_logger = logger
            self.training_loss_reduction.setup_logger(self.wandb_logger)

    def configure_model(self):
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_text: str = None,
        loss_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        output_images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output_tensor = self.module(
            images=images,
            output_images=output_images,
            input_ids=input_ids,
            position_ids=position_ids,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            labels=labels,
            inference_params=inference_params,
            input_text=input_text,
            num_image_tiles=num_image_tiles,
            image_token_mask=image_token_mask,
        )

        if torch.distributed.is_initialized():
            current_rank = torch.distributed.get_rank()
        else:
            current_rank = None
        current_rank = torch.distributed.get_rank()
        logging_condition = (
            self.trainer.training
            and current_rank == 0
            and self.trainer.global_step % 250 == 0
            and self.config.stage in ['decoder_alignment']
        )
        if logging_condition:
            if self.wandb_logger is not None:
                image_caption_embeddings = output_tensor['image_caption_embeddings']
                output_projection_embeddings = output_tensor['output_projection_embeddings']

                with torch.no_grad():
                    image_decoder = self.module.module.image_decoder

                    actual_image = image_decoder(prompt_embeds=image_caption_embeddings[:1]).images[0]
                    predicted_image = image_decoder(prompt_embeds=output_projection_embeddings[:1]).images[0]

                self.wandb_logger.log_image(
                    key="images",
                    images=[actual_image, predicted_image],
                    caption=["actual", "predicted"],
                    step=self.trainer.global_step,
                )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MimoLossReduction:
        if not self._training_loss_reduction:
            if self.config.stage in ["encoder_alignment"]:
                self._training_loss_reduction = MaskedTokenLossReduction()
            elif self.config.stage in ["decoder_alignment"]:
                self._training_loss_reduction = MimoLossReduction(generation_loss=self.config.generation_loss)
            else:
                NotImplementedError(f"Loss function not implemented for stage {self.config.stage}")

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MimoLossReduction:
        if not self._validation_loss_reduction:
            if self.config.stage in ["encoder_alignment"]:
                self._validation_loss_reduction = MaskedTokenLossReduction()
            elif self.config.stage in ["decoder_alignment"]:
                self._validation_loss_reduction = MimoLossReduction(
                    validation_step=True, generation_loss=self.config.generation_loss
                )
            else:
                NotImplementedError(f"Loss function not implemented for stage {self.config.stage}")

        return self._validation_loss_reduction
