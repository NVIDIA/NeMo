import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from nemo.lightning.megatron_parallel import MaskedTokenLossReduction


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


class MimoLossReduction(MaskedTokenLossReduction):
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True, l2_weight: float = 1.0) -> None:
        super().__init__(validation_step, val_drop_last)
        self.l2_weight = l2_weight

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        forward_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate masked token loss using superclass logic and add L2 loss.
        """
        # output_triple,new_loss_mask = forward_out
        # output, output_projection_embeddings, image_caption_embeddings = output_triple
        output_dict = forward_out

        output = output_dict['output']
        new_loss_mask = output_dict['new_loss_mask']
        output_projection_embeddings = output_dict['output_projection_embeddings']
        image_caption_embeddings = output_dict['image_caption_embeddings']
        # Use the superclass's forward method to calculate token loss
        current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        token_loss, token_loss_info = super().forward(batch={"loss_mask": new_loss_mask}, forward_out=output)
        just_token_loss = token_loss_info['avg'].clone().detach()
        # print(f"Yash loss debug rank {current_rank} just reduced_token_loss {token_loss_info['avg']}")

        l2_loss = self._calculate_l2_loss(output_projection_embeddings, image_caption_embeddings)
        l2_loss = self.l2_weight * l2_loss
        from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

        reduced_l2_loss = average_losses_across_data_parallel_group([l2_loss])

        total_loss = token_loss + l2_loss
        # logging.info(f"Yash loss debug total_loss {total_loss}")
        token_loss_info['avg'] = token_loss_info['avg'] + reduced_l2_loss
        token_loss_info.update({"l2_loss": reduced_l2_loss})

        # denoise l2 loss

        # mse_loss_weights = output_dict['denoise_mse_loss_weights']
        model_pred = output_dict['denoise_model_pred']
        target = output_dict['denoise_target']

        # gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        # gen_loss = gen_loss.mean(dim=[]) * mse_loss_weights
        # gen_loss = gen_loss.mean()
        gen_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        reduced_gen_l2_loss = average_losses_across_data_parallel_group([gen_loss])

        total_loss = total_loss + gen_loss
        token_loss_info['avg'] = token_loss_info['avg'] + reduced_gen_l2_loss
        # print(f"Rank {current_rank}: individiual l2_loss  = {l2_loss}")
        # print(f"Rank {current_rank}:reduced l2_loss  = {reduced_l2_loss}")
        # print(f"Rank {current_rank}: gen_loss  = {gen_loss}")
        # print(f"Rank {current_rank}:reduced denoise  = {reduced_gen_l2_loss}")

        # logging.info(
        #     f"Yash loss debug full loss {token_loss_info['avg'] } token_loss {token_loss} embedding loss {reduced_l2_loss} denoise loss {reduced_gen_l2_loss}"
        # )
        if current_rank == 0 or current_rank == 1:
            print(
                f"Yash rank {current_rank} avg {token_loss_info['avg'] } token_loss {token_loss} red. {just_token_loss} individual l2 {l2_loss} red. {reduced_l2_loss} denoise {gen_loss} red. {reduced_gen_l2_loss}"
            )
            print(
                f"Yash debug rank {current_rank} image_caption_embeddings mean {image_caption_embeddings.mean()} sum {image_caption_embeddings.sum()} "
            )

            print(
                f"Yash debug rank {current_rank} output_projection_embeddings mean {output_projection_embeddings.mean()} sum {output_projection_embeddings.sum()} "
            )
            print(
                f"Yash debug rank {current_rank} output_projection_embeddings first column {output_projection_embeddings[0][:,0]} "
            )
            print(
                f"Yash debug rank {current_rank} image_caption_embeddings first column {image_caption_embeddings[0][:,0]} "
            )
        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            print("******************")
            breakpoint()
        torch.distributed.barrier()
        # logging.info(
        #     f"Yash loss debug full loss {token_loss_info['avg'] } token_loss {token_loss} embe{reduced_gen_l2_loss}"
        # )
        # if torch.distributed.get_rank() == 0:
        #     wandb.log(
        #         {
        #             "full_loss": token_loss_info['avg'],
        #             "token_loss": token_loss,
        #             "embedding_l2_loss": reduced_l2_loss,
        #             "denoise_l2_loss": reduced_gen_l2_loss,
        #         }
        #     )
        return total_loss, token_loss_info

    def _calculate_l2_loss(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """Calculate L2 loss (mean squared error) between two sets of embeddings."""
        return torch.nn.functional.mse_loss(embeddings1, embeddings2)
