# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from itertools import permutations

import torch
import torch.nn.functional as F
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, LossType, NeuralType, SpectrogramType

__all__ = ["SiSNR"]


class SiSNR(Loss):
    """
    Calculate Scale-Invariant SNR
    """

    def __init__(self):
        super().__init__()

    def forward(self, target, estimate):
        """
        Args:
            target: 
                [T, B, S]
                B: Batch
                S: number of sources
            estimate:
                [T, B, S]
        """
        EPS = 1e-8
        assert target.size() == estimate.size(), f"target size {target.shape}, estimate  {estimate.shape}"
        device = estimate.device

        # look for a replacement of torch.tensor
        target_lengths = torch.tensor([estimate.shape[0]] * estimate.shape[1], device=device)

        mask = get_mask(target, target_lengths)
        estimate *= mask

        num_samples = target_lengths.contiguous().reshape(1, -1, 1).float()
        # [1, B, 1]

        mean_target = torch.sum(target, dim=0, keepdim=True) / num_samples
        mean_estimate = torch.sum(estimate, dim=0, keepdim=True) / num_samples

        zero_mean_target = target - mean_target
        zero_mean_estimate = estimate - mean_estimate

        zero_mean_target *= mask
        zero_mean_estimate *= mask

        # reshape to use broadcast
        s_target = zero_mean_target  # [T, B, C]
        s_estimate = zero_mean_estimate  # [T, B, C]
        # s_target = <s', s>s / ||s||^2
        dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
        s_target_energy = torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS  # [1, B, C]
        proj = dot * s_target / s_target_energy  # [T, B, C]
        # e_noise = s' - s_target
        e_noise = s_estimate - proj  # [T, B, C]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (torch.sum(e_noise ** 2, dim=0) + EPS)
        si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

        return -si_snr.unsqueeze(0)


def get_mask(target, target_lengths):
    """
    Args:
        target: [T, B, S]
        target_lengths: [B]
    
    Returns:
        mask: [T, B, 1]
    """
    T, B, _ = target.size()
    mask = target.new_ones((T, B, 1))
    for i in range(B):
        mask[target_lengths[i] :, i, :] = 0
    return mask


class PermuationInvarianceWrapper(Loss):
    """
    A wrapper for any existing loss to produce permutation invariance. 
    Typically referred to as permutation invariant training (PIT) in literature.

    Permutation invariance is calculated over sources/classes axis. 
    Here sources is assumed to be over last axis.
    [batch, ....., sources]
    
    ######
    #change it to zero axis later
    #####

    Args:
        base_loss: torch.nn.Module

    Returns:
        pi_loss: torch.nn.Module
    """

    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, preds, targets):
        """
        Args:
            preds: torch.Tensor
                [Batch, ...., sources]
            targets: torch.Tensor
                [Batch, ...., sources]            
        
        Returns:
            pi_loss: torch.Tensor
                [batch]
                permutation invariant loss for current batch
            
            optimal_perm: list
                optimal assignment of predictions and targets

        """
        losses = []
        optimal_perm = []

        # go through each sample in batch
        for p, t in zip(preds, targets):
            loss, order = self._optimal_permutation_loss(p, t)
            optimal_perm.append(order)
            losses.append(loss)
        pi_loss = torch.stack(losses)
        return pi_loss, optimal_perm

    def _optimal_permutation_loss(self, pred, target):
        """
        Args:
            pred: torch.Tensor
                [...., sources]
            target: torch.Tensor
                [...., sources]
        
        Return:
            loss: torch.Tensor
                [1]
                permutation invariant loss for current 

            assignment_order: tuple
                which targets are assigned to which sources
        """

        num_sources = target.size(-1)

        # expand pred and target tensor
        pred = pred.unsqueeze(-2).repeat(*[1 for x in range(len(pred.shape) - 1)], num_sources, 1)
        target = target.unsqueeze(-1).repeat(1, *[1 for x in range(len(target.shape) - 1)], num_sources)

        # loss for all possible permutations
        loss_matrix = self.base_loss(pred, target)

        # mean over all dims except last two
        mean_over_dims = [x for x in range(len(loss_matrix.shape) - 2)]

        loss_matrix = loss_matrix.mean(dim=mean_over_dims)

        loss, optimal_perm = self._optimal_perm_assigment(loss_matrix)
        return loss, optimal_perm

    def _optimal_perm_assigment(self, loss_matrix):
        """
        Args:
            loss_matrix: torch.Tensor
                [sources, sources]
        
        Returns:
            loss: torch.Tensor
                [1]
                loss corresponding to optimal assignment
            optimal_perm: tuple
                [sources]
                optimal assignment which minimizes the loss
        """
        num_sources = loss_matrix.shape[0]
        loss = None
        optimal_perm = None
        for p in permutations(range(num_sources)):
            current_loss = loss_matrix[range(num_sources), p].mean()

            if loss is None or loss > current_loss:
                loss = current_loss
                optimal_perm = p
        return loss, optimal_perm
