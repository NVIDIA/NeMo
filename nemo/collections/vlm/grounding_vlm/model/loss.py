import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.collections.vlm.grounding_vlm.model.bbox_losses import HungarianMatchingLoss

class GroundedVLMLossReduction(MaskedTokenLossReduction):
    '''
    Loss function containing following losses:
    - token loss
    - classification loss (BCE loss with sigmoid)
    - detection loss (hungarian loss with giou)
    - semantic segmentation loss (focal loss)
    - instance segmentation loss (hungarian loss with giou - TODO)
    '''
    def __init__(
        self,
        validation_step: bool = False,
        val_drop_last: bool = True,
        cls_loss_weight: float = 1.0,
        instance_det_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(validation_step, val_drop_last)
        self.cls_loss_weight = cls_loss_weight
        self.instance_det_loss_weight = instance_det_loss_weight
        self.logger = None
        self.hungarian_matching_loss = HungarianMatchingLoss()

    def setup_logger(self, logger):
        self.logger = logger

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        forward_out: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate masked token loss using superclass logic and add L2 loss.
        """

        output = forward_out['token_loss']
        final_loss_mask = forward_out['final_loss_mask']

        # token loss
        token_loss, num_valid_tokens, token_loss_info = super().forward(batch={"loss_mask": final_loss_mask}, forward_out=output)
        num_valid_tokens = int(num_valid_tokens.item()) if isinstance(num_valid_tokens, torch.Tensor) else num_valid_tokens
        # compute average token loss (legacy)
        token_loss_info['avg'] = token_loss / max(1, num_valid_tokens)
        just_token_loss = token_loss_info['avg'].clone().detach()

        total_loss = token_loss / max(1, num_valid_tokens)

        # classification loss
        cls_logits = forward_out['cls_logits']    
        cls_labels = batch['cls_labels']
        cls_loss_mask = batch['cls_loss_mask']
        cls_loss = torch.tensor(0.0, device=cls_logits.device)

        if cls_logits is not None:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_labels.to(cls_logits.dtype), weight=cls_loss_mask.to(cls_logits.dtype), reduction="mean")
            cls_loss = self.cls_loss_weight * cls_loss
            reduced_cls_loss = average_losses_across_data_parallel_group([cls_loss])

        total_loss = total_loss + cls_loss

        # detection loss
        gt_quads = batch['instance_det_ids']  # [cu_num_instances, 4]
        instance_cu_seqlen = batch['instance_cu_seqlen']
        pred_quads = forward_out['instance_logits'] # [cu_num_instances, 4]
        
        instance_det_loss = torch.tensor(0.0, device=pred_quads.device)

        if pred_quads is not None:
            instance_det_loss = self._hungarian_matching_loss(gt_quads, pred_quads, instance_cu_seqlen)
            instance_det_loss = self.instance_det_loss_weight * instance_det_loss
            reduced_instance_det_loss = average_losses_across_data_parallel_group([instance_det_loss])

        total_loss = total_loss + instance_det_loss

        token_loss_info['avg'] = token_loss_info['avg'] + reduced_cls_loss + reduced_instance_det_loss
        token_loss_info.update({"cls_loss": reduced_cls_loss})
        token_loss_info.update({"instance_det_loss": reduced_instance_det_loss})

        logging.info(f"token_loss: {just_token_loss}, cls_loss: {reduced_cls_loss}, instance_det_loss: {reduced_instance_det_loss}")

        if self.logger:
            self.logger.log_metrics({'token_loss': just_token_loss})
            self.logger.log_metrics({'cls_loss': reduced_cls_loss})
            self.logger.log_metrics({'instance_det_loss': reduced_instance_det_loss})

        return total_loss, token_loss_info

    def _hungarian_matching_loss(self, gt_quads: torch.Tensor, pred_quads: torch.Tensor, instance_cu_seqlen: torch.Tensor, loss_only: bool = True) -> torch.Tensor:
        """
        Calculate hungarian matching loss between gt_quads and pred_quads.
        """
        loss, metadata = self.hungarian_matching_loss(gt_quads, pred_quads, instance_cu_seqlen)
        if loss_only:
            return loss
        return loss, metadata