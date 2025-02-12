import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction


class MimoLossReduction(MaskedTokenLossReduction):

    def __init__(
        self,
        validation_step: bool = False,
        val_drop_last: bool = True,
        l2_weight: float = 1.0,
        generation_loss=False,
    ) -> None:
        super().__init__(validation_step, val_drop_last)
        self.l2_weight = l2_weight
        self.generation_loss = generation_loss
        self.logger = None

    def setup_logger(self, logger):
        self.logger = logger

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        forward_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate masked token loss using superclass logic and add L2 loss.
        """

        output_dict = forward_out

        output = output_dict['output']
        new_loss_mask = output_dict['new_loss_mask']
        output_projection_embeddings = output_dict['output_projection_embeddings']
        image_caption_embeddings = output_dict['image_caption_embeddings']

        # token loss
        token_loss, token_loss_info = super().forward(batch={"loss_mask": new_loss_mask}, forward_out=output)
        just_token_loss = token_loss_info['avg'].clone().detach()

        total_loss = token_loss
        # L2 loss
        l2_loss = self._calculate_l2_loss(output_projection_embeddings, image_caption_embeddings)
        l2_loss = self.l2_weight * l2_loss
        reduced_l2_loss = average_losses_across_data_parallel_group([l2_loss])

        total_loss = total_loss + l2_loss
        token_loss_info['avg'] = token_loss_info['avg'] + reduced_l2_loss
        token_loss_info.update({"l2_loss": reduced_l2_loss})

        gen_loss = None
        # denoise loss
        if self.generation_loss:
            assert model_pred is not None
            assert target is not None

            model_pred = output_dict['denoise_model_pred']
            target = output_dict['denoise_target']
            gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            reduced_gen_l2_loss = average_losses_across_data_parallel_group([gen_loss])

            total_loss = total_loss + gen_loss
            token_loss_info['avg'] = token_loss_info['avg'] + reduced_gen_l2_loss
            logging.info(f"token_loss: {just_token_loss}, l2_loss: {reduced_l2_loss}, gen_loss: {reduced_gen_l2_loss}")
        else:
            logging.info(f"token_loss: {just_token_loss}, l2_loss: {reduced_l2_loss}")
        if self.logger:
            self.logger.log_metrics({'token_loss': just_token_loss})
            self.logger.log_metrics({'l2_loss': l2_loss})
            if gen_loss:
                self.logger.log_metrics({'gen_loss': reduced_gen_l2_loss})
        return total_loss, token_loss_info

    def _calculate_l2_loss(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """Calculate L2 loss (mean squared error) between two sets of embeddings."""
        return torch.nn.functional.mse_loss(embeddings1, embeddings2)
