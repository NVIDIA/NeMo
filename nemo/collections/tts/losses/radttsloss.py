###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import torch
import torch.nn as nn
from torch.nn import functional as F
from nemo.collections.tts.helpers.common import get_mask_from_lengths


class FeaturePredictionLoss(torch.nn.Module):
    def __init__(self, model_config, loss_weight, sigma=1.0, gm_loss=False,
                 mask_unvoiced_f0=False):
        super(FeaturePredictionLoss, self).__init__()
        self.sigma = sigma
        self.model_name = model_config['name']
        self.loss_weight = loss_weight
        if 'n_group_size' in model_config['hparams']:
            self.n_group_size = model_config['hparams']['n_group_size']
        else:
            self.n_group_size = 1
        self.mask_unvoiced_f0 = mask_unvoiced_f0

    def compute_loss(self, z, log_det_W_list, log_s_list, n_elements, n_dims,
                     mask, means=None, variances=None):
        log_det_W_total = torch.zeros(1).cuda()
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s * mask)
                if len(log_det_W_list):
                    log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s * mask)
                if len(log_det_W_list):
                    log_det_W_total += log_det_W_list[i]

        log_det_W_total *= n_elements

        z = z * mask
        if means is None:
            prior_NLL = torch.sum(z*z)/(2*self.sigma*self.sigma)
        else:
            prior_NLL = 0.5*((z - means) ** 2)/(self.sigma*self.sigma)
            prior_NLL = (prior_NLL*mask).sum()

        loss = prior_NLL - log_s_total - log_det_W_total

        denom = n_elements * n_dims
        loss = loss / denom
        loss_prior = prior_NLL / denom
        return loss, loss_prior

    def forward(self, model_output, in_lens, with_stft_loss=False):
        loss = torch.zeros(1).cuda()
        loss_prior = torch.zeros(1).cuda()
        loss_stft = torch.zeros(1).cuda()
        mask = get_mask_from_lengths(in_lens // self.n_group_size)
        if 'x_hat' in model_output:
            x_hat = model_output['x_hat']
            x = model_output['x']
            x = x[:, None] if len(x.shape) == 2 else x
            x_hat = x_hat * mask[:, None]
            if x_hat.shape[1] == 1:
                loss = F.mse_loss(x_hat, x, reduction='sum')
                loss = loss / mask.sum()
            elif x_hat.shape[1] == 2:
                fev, fev_hat = x[:, 0:1], x_hat[:, 0:1]
                energy, energy_hat = x[:, 1:2], x_hat[:, 1:2]
                loss = F.mse_loss(fev_hat, fev, reduction='sum')
                loss_prior = F.mse_loss(energy_hat, energy, reduction='sum')
                # massive hack for plotting
                loss = loss / mask.sum()
                loss_prior = loss_prior / mask.sum()
            elif x_hat.shape[1] == 3:
                f0 = x[:, 0:1]
                energy = x[:, 1:2]
                voiced = x[:, 2:]
                f0_hat = x_hat[:, 0:1]
                energy_hat = x_hat[:, 1:2]
                voiced_hat = x_hat[:, 2:]
                # ueber massive hack for plotting
                # fev
                if self.mask_unvoiced_f0:
                    loss = F.mse_loss(f0_hat*voiced, f0*voiced, reduction='sum')
                    loss = loss / voiced.sum()
                else:
                    loss = F.mse_loss(f0_hat, f0, reduction='sum')
                    loss = loss / mask.sum()
                # ENERGY
                loss_prior = F.mse_loss(energy_hat, energy, reduction='sum')
                loss_prior = loss_prior / mask.sum()
                loss += loss_prior
                # VOICED
                loss_stft = F.binary_cross_entropy_with_logits(
                    voiced_hat, voiced, reduction='sum')
                loss_stft = loss_stft / mask.sum()
                loss += loss_stft
                with_stft_loss = False

            if with_stft_loss:
                x_mag = self.stft(x[:, 0])
                x_hat_mag = self.stft(x_hat[:, 0])
                loss_stft = F.l1_loss(x_hat_mag, x_mag, reduction='sum')
                loss_stft = loss_stft / mask.sum()
        elif 'z' in model_output and len(model_output):
            n_elements = in_lens.sum() // self.n_group_size
            mask = mask[:, None].float()
            n_dims = model_output['z'].size(1)
            loss, loss_prior = self.compute_loss(
                model_output['z'],
                model_output['log_det_W_list'],
                model_output['log_s_list'], n_elements, n_dims, mask)

        return loss, loss_prior, loss_stft


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0),
            value=self.blank_logprob)
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid]+1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                :query_lens[bid], :, :key_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(curr_logprob, target_seq,
                                    input_lengths=query_lens[bid:bid+1],
                                    target_lengths=key_lens[bid:bid+1])
            cost_total += ctc_cost
        cost = cost_total/attn_logprob.shape[0]
        return cost


class RadTTSLoss(torch.nn.Module):
    def __init__(self, sigma=1.0, gm_loss=False, dur_loss_weight=0.0,
                 feat_loss_weight=1.0, n_group_size=1,
                 duration_model_config=None, feature_model_config=None,
                 mask_unvoiced_f0=False):
        super(RadTTSLoss, self).__init__()
        self.sigma = sigma
        self.dur_loss_weight = dur_loss_weight
        self.n_group_size = n_group_size
        self.attn_ctc_loss = AttentionCTCLoss()
        self.dur_pred_loss = FeaturePredictionLoss(
            duration_model_config, dur_loss_weight,
            mask_unvoiced_f0=False)
        self.feat_pred_loss = FeaturePredictionLoss(
            feature_model_config, feat_loss_weight,
            mask_unvoiced_f0=mask_unvoiced_f0)

    def compute_loss(self, z, log_det_W_list, log_s_list, n_elements, n_dims,
                     mask, means=None, variances=None):
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s * mask)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s * mask)
                log_det_W_total += log_det_W_list[i]

        log_det_W_total *= n_elements

        z = z * mask
        if means is None:
            prior_NLL = torch.sum(z*z)/(2*self.sigma*self.sigma)
        else:
            prior_NLL = 0.5*((z - means) ** 2)/(self.sigma*self.sigma)
            prior_NLL = (prior_NLL*mask).sum()

        loss = prior_NLL - log_s_total - log_det_W_total

        denom = n_elements * n_dims
        loss = loss / denom
        loss_prior = prior_NLL / denom
        return loss, loss_prior

    def forward(self, model_output, in_lens, out_lens):
        loss_mel, loss_duration = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        loss_fev = torch.zeros(1).cuda()
        loss_fev_extra = torch.zeros(1).cuda()
        loss_prior_mel = torch.zeros(1).cuda()
        loss_prior_duration = torch.zeros(1).cuda()
        loss_prior_fev = torch.zeros(1).cuda()

        if len(model_output['z_mel']):
            n_elements = out_lens.sum() // self.n_group_size
            mask = get_mask_from_lengths(out_lens // self.n_group_size)
            mask = mask[:, None].float()
            n_dims = model_output['z_mel'].size(1)
            means = variances = None
            if model_output['text_conditional_priors'] is not None:
                means = model_output['text_conditional_priors']
                variances = None
            loss_mel, loss_prior_mel = self.compute_loss(
                model_output['z_mel'], model_output['log_det_W_list'],
                model_output['log_s_list'], n_elements, n_dims, mask,
                means, variances)

        if model_output['duration_model_outputs'] is not None:
            loss_duration, loss_prior_duration, _ = self.dur_pred_loss(
                model_output['duration_model_outputs'], in_lens)

        if model_output['feature_model_outputs'] is not None:
            loss_fev, loss_prior_fev, loss_fev_extra = self.feat_pred_loss(
                model_output['feature_model_outputs'], out_lens)

        # compute the CTC alignment cost for the attention
        ctc_cost = self.attn_ctc_loss(
            model_output['attn_logprob'], in_lens, out_lens)

        return {'loss_mel': loss_mel,
                'loss_prior_mel': loss_prior_mel,
                'loss_duration': loss_duration,
                'loss_prior_duration': loss_prior_duration,
                'loss_fev': loss_fev,
                'loss_prior_fev': loss_prior_fev,
                'loss_fev_extra': loss_fev_extra,
                'loss_ctc': ctc_cost
                }  # NLL wrt. gaussian prior


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(soft_attention[hard_attention == 1]).sum()
        return -log_sum / hard_attention.sum()
