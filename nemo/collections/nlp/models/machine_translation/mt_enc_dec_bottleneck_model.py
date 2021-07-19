# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import json
import random
from multiprocessing import Value
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as pt_data
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu

from nemo.collections.common.losses import NLLLoss
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTBottleneckModelConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.modules.common.transformer import AttentionBridge, TopKSequenceGenerator
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils

__all__ = ['MTBottleneckModel']


class MTBottleneckModel(MTEncDecModel):
    """
    Machine translation model which supports bottleneck architecture,
    and VAE and MIM loss.

    See MIM loss in <https://arxiv.org/pdf/2003.02645.pdf>
    """

    def __init__(self, cfg: MTBottleneckModelConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        recon_per_token: bool = True

        self.model_type: str = cfg.get("model_type", "seq2seq-br")
        self.min_logv: float = cfg.get("min_logv", -6)
        self.ortho_loss_coef: float = cfg.get("ortho_loss_coef", 0.0)
        self.att_bridge_size: int = cfg.get("att_bridge_size", 512)
        self.att_bridge_k: int = cfg.get("att_bridge_k", 16)
        self.att_bridge_inner_size: int = cfg.get("att_bridge_inner_size", 1024)
        self.non_recon_warmup_batches: int = cfg.get("non_recon_warmup_batches", 200000)
        self.recon_per_token: bool = cfg.get("recon_per_token", True)

        # TODO: add support in label smoothing for per-sample reconstruction loss
        if not self.recon_per_token:
            loss_fn = NLLLoss(ignore_index=self.decoder_tokenizer.pad_id, reduction='none',)
            self.loss_fn = self.eval_loss_fn = loss_fn

        if self.model_type not in ["seq2seq", "seq2seq-br", "seq2seq-mim", "seq2seq-vae"]:
            raise ValueError("Unknown model_type = {model_type}".format(model_type=self.model_type,))

        if self.model_type != "seq2seq":
            # project bridge dimension back to decoder hidden dimensions
            if self.att_bridge_size != self.encoder.hidden_size:
                self.latent2hidden = torch.nn.Linear(self.att_bridge_size, self.encoder.hidden_size)
            else:
                self.latent2hidden = torch.nn.Identity()

            self.att_bridge = AttentionBridge(
                hidden_size=self.encoder.hidden_size, k=self.att_bridge_k, bridge_size=self.att_bridge_size,
            )

            # project dimension of encoder hidden to bridge dimension
            if self.encoder.hidden_size != self.att_bridge_size:
                self.hidden2latent_mean = torch.nn.Linear(self.encoder.hidden_size, self.att_bridge_size)
            else:
                self.hidden2latent_mean = torch.nn.Identity()

            # for probabilistic latent variable models we also need variance
            if self.model_type in ["seq2seq-mim", "seq2seq-vae"]:
                if self.encoder.hidden_size != self.att_bridge_size:
                    self.hidden2latent_logv = torch.nn.Linear(self.encoder.hidden_size, self.att_bridge_size)
                else:
                    self.hidden2latent_logv = torch.nn.Identity()
        else:
            # seq2seq
            self.latent2hidden = torch.nn.Identity()

    def eval_epoch_end(self, outputs, mode):
        # call parent for logging
        super().eval_epoch_end(outputs, mode)

        # if user specifies one validation dataloader, then PTL reverts to giving a list of dictionary instead of a list of list of dictionary
        if isinstance(outputs[0], dict):
            outputs = [outputs]

        for dataloader_idx, output in enumerate(outputs):
            # add logs if available in outputs
            log_dict = {}
            for x in output:
                if "log" in x:
                    for k, v in x["log"].items():
                        log_dict[k] = log_dict.get(k, []) + [v]

            for k, v in log_dict.items():
                if dataloader_idx == 0:
                    self.log(f"{mode}_{k}", np.mean(v), sync_dist=True)
                else:
                    self.log(f"{mode}_{k}_dl_index_{dataloader_idx}", np.mean(v), sync_dist=True)

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []

        return result

    def sample_z(self, hidden, hidden_mask, return_ortho_loss=False):
        """
        Sample latent code z with reparameterization from bridge for
        probabilistic latent variable models, or return value for
        probabilistic models (seq2seq, seq2seq-br)
        """
        if self.model_type == "seq2seq":
            # seq2seq
            z = z_mean = hidden
            z_logv = torch.zeros_like(hidden)
            z_mask = hidden_mask
            ortho_loss = 0.0
        else:
            # seq2seq-br, seq2seq-mim, seq2seq-vae

            # project hidden to a fixed size bridge using k attention heads
            res = self.att_bridge(hidden=hidden, hidden_mask=hidden_mask, return_ortho_loss=return_ortho_loss,)

            if return_ortho_loss:
                bridge_hidden, ortho_loss = res
            else:
                bridge_hidden = res

            # all bottleneck models have mean
            z_mean = self.hidden2latent_mean(bridge_hidden)

            # sample posterior q(z|x) for MIM and VAE
            if self.model_type in ["seq2seq-mim", "seq2seq-vae"]:
                z_logv = self.hidden2latent_logv(bridge_hidden)
                # avoid numerical instability for MIM
                z_logv = z_logv.clamp_min(self.min_logv)
                # sample z with reparameterization
                e = torch.randn_like(z_mean)
                z = e * torch.exp(0.5 * z_logv) + z_mean
            else:
                z_logv = torch.zeros_like(z_mean)
                z = z_mean

            # all steps in bottleneck bridge are used
            z_mask = torch.ones(z.shape[0:2]).to(hidden_mask)

        if return_ortho_loss:
            return z, z_mean, z_logv, z_mask, ortho_loss
        else:
            return z, z_mean, z_logv, z_mask

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask, labels, train=True, return_info=False):
        """
        return_info - if True, returns loss, info_dict with additional information
                      regarding the loss that can be logged
        """
        info_dict = {}

        src_hiddens = self.encoder(input_ids=src, encoder_mask=src_mask,)

        # build posterior distribution q(x|z)
        z, z_mean, z_logv, bridge_mask, ortho_loss = self.sample_z(
            hidden=src_hiddens,
            hidden_mask=src_mask,
            # we always return return_ortho_loss here even if ignored
            # to avoid recomputing attention bridge twice
            return_ortho_loss=True,
        )

        # build decoding distribution
        bridge_hiddens_dec = self.latent2hidden(z)

        tgt_hiddens = self.decoder(
            input_ids=tgt, decoder_mask=tgt_mask, encoder_embeddings=bridge_hiddens_dec, encoder_mask=bridge_mask,
        )

        log_probs = self.log_softmax(hidden_states=tgt_hiddens)

        recon_loss_fn = self.loss_fn if train else self.eval_loss_fn

        if self.recon_per_token:
            log_p_x_given_z_per_token = -recon_loss_fn(log_probs=log_probs, labels=labels)

            log_p_x_given_z = log_p_x_given_z_per_token
            log_p_x_given_z_per_token = log_p_x_given_z_per_token.detach()
        else:
            # averaging of log_p_x_given_z per sample
            output_mask = (labels != self.decoder_tokenizer.pad_id).type_as(log_probs)

            log_p_x_given_z_per_token = (
                -recon_loss_fn(log_probs=log_probs, labels=labels,).view(log_probs.shape[:2]) * output_mask
            )

            # probability per sample
            log_p_x_given_z = log_p_x_given_z_per_token.sum(-1).mean()

            tokens = output_mask.sum()
            log_p_x_given_z_per_token = log_p_x_given_z_per_token.sum().detach() / tokens

            info_dict["log_p_x_given_z"] = log_p_x_given_z.detach().cpu()

        info_dict["log_p_x_given_z_per_token"] = log_p_x_given_z_per_token.detach().cpu()

        # loss warmup during training only
        if train:
            trainer = self.trainer
            # if we do not have a trainer ignore annealing
            if trainer is None:
                # ignore warmup and auxiliary loss
                warmup_coef = 1.0
                ortho_loss_coef = 0.0
            else:
                global_step = self.trainer.global_step

                warmup_coef = min(global_step / self.non_recon_warmup_batches, 1)
                ortho_loss_coef = self.ortho_loss_coef
        else:
            # ignore warmup and auxiliary loss
            warmup_coef = 1.0
            ortho_loss_coef = 0.0

        info_dict["warmup_coef"] = warmup_coef

        if self.model_type in ["seq2seq-mim", "seq2seq-vae"]:
            # tokens = tgt_mask.sum()
            q_z_given_x = torch.distributions.Normal(loc=z_mean, scale=torch.exp(0.5 * z_logv),)
            # average latent distribution to match averaging of observations
            if self.recon_per_token:
                # TODO: replace mean with division by number of tokens
                log_q_z_given_x = q_z_given_x.log_prob(z).mean(-1).mean(-1).mean()
            else:
                log_q_z_given_x = q_z_given_x.log_prob(z).sum(-1).sum(-1).mean()

            # build prior distribution
            p_z = torch.distributions.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z),)
            # average latent distribution to match averaging of observations
            if self.recon_per_token:
                # TODO: replace mean with division by number of tokens
                log_p_z = p_z.log_prob(z).mean(-1).mean(-1).mean()
            else:
                log_p_z = p_z.log_prob(z).sum(-1).sum(-1).mean()

            if self.model_type == "seq2seq-mim":
                loss_terms = 0.5 * (log_q_z_given_x + log_p_z)
            elif self.model_type == "seq2seq-vae":
                # KL divergence -Dkl( q(z|x) || p(z) )
                loss_terms = log_p_z - log_q_z_given_x

            # show loss value for reconstruction but train with MIM/VAE loss
            computed_loss = log_p_x_given_z + warmup_coef * loss_terms
            display_loss = log_p_x_given_z_per_token

            info_dict["log_q_z_given_x"] = log_q_z_given_x.detach().cpu()
            info_dict["log_p_z"] = log_p_z.detach().cpu()
            info_dict["kl_div_q_p"] = (log_q_z_given_x - log_p_z).detach().cpu()

        elif self.model_type in ["seq2seq", "seq2seq-br"]:
            computed_loss = log_p_x_given_z
            display_loss = log_p_x_given_z_per_token

        loss = -((computed_loss - computed_loss.detach()) + display_loss)

        # add attention orthogonality loss
        loss = loss + warmup_coef * ortho_loss_coef * ortho_loss

        info_dict["computed_loss"] = -computed_loss.detach().cpu()
        if torch.is_tensor(ortho_loss):
            ortho_loss = ortho_loss.detach().cpu()
        info_dict["ortho_loss"] = ortho_loss

        if return_info:
            return loss, info_dict
        else:
            return loss

    @torch.no_grad()
    def batch_translate(
        self, src: torch.LongTensor, src_mask: torch.LongTensor,
    ):
        """
        Translates a minibatch of inputs from source language to target language.
        Args:
            src: minibatch of inputs in the src language (batch x seq_len)
            src_mask: mask tensor indicating elements to be ignored (batch x seq_len)
        Returns:
            translations: a list strings containing detokenized translations
            inputs: a list of string containing detokenized inputs
        """
        mode = self.training
        try:
            self.eval()
            src_hiddens = self.encoder(input_ids=src, encoder_mask=src_mask)

            z, _, _, bridge_mask = self.sample_z(
                hidden=src_hiddens,
                hidden_mask=src_mask,
                # we return return_ortho_loss only during training
                return_ortho_loss=False,
            )
            bridge_hiddens_dec = self.latent2hidden(z)

            beam_results = self.beam_search(encoder_hidden_states=bridge_hiddens_dec, encoder_input_mask=bridge_mask)

            beam_results = self.filter_predicted_ids(beam_results)

            translations = [self.decoder_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
            inputs = [self.encoder_tokenizer.ids_to_text(inp) for inp in src.cpu().numpy()]
            if self.target_processor is not None:
                translations = [
                    self.target_processor.detokenize(translation.split(' ')) for translation in translations
                ]

            if self.source_processor is not None:
                inputs = [self.source_processor.detokenize(item.split(' ')) for item in inputs]
        finally:
            self.train(mode=mode)

        return inputs, translations

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        train_loss, info_dict = self(src_ids, src_mask, tgt_ids, tgt_mask, labels, train=True, return_info=True)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
        }
        tensorboard_logs.update(info_dict)

        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode, dataloader_idx=0):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)

        if self.multilingual:
            self.source_processor = self.source_processor_list[dataloader_idx]
            self.target_processor = self.target_processor_list[dataloader_idx]

        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        eval_loss, info_dict = self(src_ids, src_mask, tgt_ids, tgt_mask, labels, train=False, return_info=True)
        # this will run encoder twice -- TODO: potentially fix
        _, translations = self.batch_translate(src=src_ids, src_mask=src_mask)

        # TODO: log info_dict similar to train_step
        num_measurements = labels.shape[0] * labels.shape[1]
        if dataloader_idx == 0:
            getattr(self, f'{mode}_loss')(
                loss=eval_loss, num_measurements=num_measurements,
            )
        else:
            getattr(self, f'{mode}_loss_{dataloader_idx}')(
                loss=eval_loss, num_measurements=num_measurements,
            )
        np_tgt = tgt_ids.detach().cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        ground_truths = [self.target_processor.detokenize(tgt.split(' ')) for tgt in ground_truths]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()
        return {
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
            'log': {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in info_dict.items()},
        }
