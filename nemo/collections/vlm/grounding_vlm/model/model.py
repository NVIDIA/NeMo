from nemo.collections.vlm.grounding_vlm.model.base import Qwen2VLGroundingConfig
from typing import Callable, Dict, Optional

import lightning.pytorch as L
import torch
from megatron.core.inference_params import InferenceParams
from megatron.core.optimizer import OptimizerConfig
from torch import nn

from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.lightning.megatron_parallel import MaskedTokenLossReductionWithLossMask
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.collections.vlm.neva.model.base import NevaModel
from nemo.collections.vlm.grounding_vlm.model.loss import GroundedVLMLossReduction


class Qwen2GroundingVLModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    """Lightning Wrapper for Qwen2VLGrounding Model"""

    def __init__(
        self,
        config: Qwen2VLGroundingConfig,
        model_version: str,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        # pylint: disable=C0115,C0116
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = GroundedVLMLossReduction()
        self._validation_loss_reduction = GroundedVLMLossReduction(validation_step=True)
        self.model_version = model_version
        assert self.model_version in ["qwen2-vl", "qwen25-vl"], "model_version only supports qwen2-vl and qwen25-vl."

    def configure_model(self, vp_stage: Optional[int] = None) -> None:
        # pylint: disable=C0115,C0116
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer, vp_stage=vp_stage)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.FloatTensor] = None,
        # Grounding VLM specific arguments
        cls_token_ids: Optional[torch.Tensor] = None,
        cls_attention_mask: Optional[torch.Tensor] = None,
        cls_labels: Optional[torch.Tensor] = None,
        cls_loss_mask: Optional[torch.Tensor] = None,
        # detection params
        instance_det_ids: Optional[torch.Tensor] = None,
        instance_cu_seqlen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        output_tensor = self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=loss_mask,
            labels=labels,
            inference_params=inference_params,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            # Grounding VLM specific arguments
            cls_token_ids=cls_token_ids,
            cls_attention_mask=cls_attention_mask,
            cls_labels=cls_labels,
            cls_loss_mask=cls_loss_mask,
            instance_det_ids=instance_det_ids,
            instance_cu_seqlen=instance_cu_seqlen,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        # pylint: disable=C0115,C0116
        return self.config.data_step_fn(dataloader_iter, self.model_version)

    def forward_step(self, batch) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # pylint: disable=C0115,C0116
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._training_loss_reduction:
            self._training_loss_reduction = GroundedVLMLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReductionWithLossMask:
        # pylint: disable=C0115,C0116
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = GroundedVLMLossReduction(validation_step=True)

        return self._validation_loss_reduction
