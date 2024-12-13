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

import copy
import os
import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from lhotse.dataset.collation import collate_vectors, collate_matrices

from nemo.collections.asr.data.audio_to_text_lhotse_speaker import LhotseSpeechToTextSpkBpeDataset, LhotseSpeechToTextMSpkBpeDataset

from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType


from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.eesd_models import SortformerEncLabelModel
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo

from nemo.utils import logging

class EncDecRNNTBPEMITSASRModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        if 'diar_model_path' in self.cfg:
            self._init_diar_model()
            self.num_speakers = cfg.model_defaults.get('num_speakers', 4)
            self.diar_kernel_type = cfg.get('diar_kernel', None)
            self.binary_diar_preds = cfg.get('binary_diar_preds', True)
            self.spk_supervision = cfg.get('spk_supervision', 'rttm')

            self.max_query_len = cfg.get('max_query_len', 128)
            self.min_query_len = cfg.get('min_query_len', 12)
            self.query_embedding_type = cfg.get('query_embedding_type', 'asr')
            self.fixed_query_len = cfg.get('fixed_query_len', True)
            self.continuous_query = cfg.get('continuous_query', True)
            self.overlapping_query = cfg.get('overlapping_query', False)

            if self.diar_kernel_type == 'metacat':
                # projection layer
                proj_in_size = self.num_speakers * cfg.model_defaults.enc_hidden
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj
            elif self.diar_kernel_type == 'metacat_residule':
                # projection layer
                proj_in_size = cfg.model_defaults.enc_hidden
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj

    def _init_diar_model(self):
        """
        Initialize the speaker model.
        """
        logging.info(f"Initializing diarization model from pretrained checkpoint {self.cfg.diar_model_path}")

        model_path = self.cfg.diar_model_path
        # model_path = '/home/jinhanw/workdir/workdir_nemo_speaker_asr/dataloader/pipeline/checkpoints/sortformer/im303a-ft7_epoch6-19.nemo'

        if model_path.endswith('.nemo'):
            pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        else:
            pretrained_diar_model = None
            logging.info("Model path incorrect")

        self.diarization_model = pretrained_diar_model

        if self.cfg.freeze_diar:
           self.diarization_model.eval()

    def forward_diar(
        self,
        audio_signal=None,
        audio_signal_length=None
    ):
        with torch.no_grad():
            processed_signal, processed_signal_length = self.diarization_model.process_signal(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
            processed_signal = processed_signal[:, :, :processed_signal_length.max()]
            pre_encode, pre_encode_length = self.diarization_model.encoder.pre_encode(x=processed_signal.transpose(1, 2), lengths=processed_signal_length)
            emb_seq, _ = self.diarization_model.frontend_encoder(processed_signal=pre_encode, processed_signal_length=pre_encode_length, pre_encode_input=True)
            preds = self.diarization_model.forward_infer(emb_seq)

        return preds, pre_encode, pre_encode_length
    
    def forward_pre_encode(
        self,
        input_signal=None,
        input_signal_length=None
    ): 
        processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        processed_signal, processed_signal_length = self.encoder.pre_encode(x=processed_signal.transpose(1, 2).contiguous(), lengths=processed_signal_length)

        return processed_signal, processed_signal_length
    
    def forward_diar_kernel(
        self,
        encoded,
        encoded_len,
        diar_preds
    ):
        if self.diar_kernel_type == 'metacat_residule':
            concat_enc_states = encoded * diar_preds.unsqueeze(1)
            encoded = encoded + self.joint_proj(concat_enc_states.transpose(1, 2)).transpose(1, 2)
        
        return encoded

    def forward_train_val(
        self, 
        signal=None,
        signal_len=None,
        query=None,
        query_len=None,
        transcript=None,
        transcript_len=None,
        spk_targets=None,
        query_speaker_ids=None
        ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tpensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        # Step 1: get pre-encoded embeddings from asr pre-encoder and diar pre-encoder
        # also get diar predictions

        # Step 1: Concatenate query and signal
        if query.shape[1] < signal.shape[1]:
            query = F.pad(query, (0, signal.shape[1] - query.shape[1]), value=0)
        else:
            signal = F.pad(signal, (0, query.shape[1] - signal.shape[1]), value=0)

        query_signal = torch.cat([query, signal], dim=0) # 2B x T
        query_signal_len = torch.cat([query_len, signal_len], dim=0)
        
        # Step 2: get pre-encoded embeddings from asr pre-encoder and diar pre-encoder, here we get query embeddings from another query audio
        asr_pre_encoded, asr_pre_encoded_len = self.forward_pre_encode(input_signal=query_signal, input_signal_length=query_signal_len)
        diar_preds, diar_pre_encoded, diar_pre_encoded_len = self.forward_diar(audio_signal=query_signal, audio_signal_length=query_signal_len)

        query_pre_encoded1, query_pre_encoded_len1 = asr_pre_encoded[:query.size(0)], asr_pre_encoded_len[:query.size(0)]
        target_pre_encoded, target_pre_encoded_len = asr_pre_encoded[query.size(0):], asr_pre_encoded_len[query.size(0):]
        diar_preds = diar_preds[query.size(0):]

        # Choose which embeddings to use for query
        if self.query_embedding_type == 'asr':
            pass
        elif self.query_embedding_type == 'diar':
            query_pre_encoded1, query_pre_encoded_len1 = diar_pre_encoded[:query.size(0)], diar_pre_encoded_len[:query.size(0)]
            target_pre_encoded, target_pre_encoded_len = diar_pre_encoded[query.size(0):], diar_pre_encoded_len[query.size(0):]
        else:
            raise ValueError(f"Unknown query_embedding_type: {self.query_embedding_type}, should be 'asr' or 'diar'")

        # Step 3: find the query embeddings from the pre-encoded embeddings based on diar predictions, here we get query embeddings from the signal itself
        if self.spk_supervision == 'rttm':
            targets = spk_targets
        elif self.spk_supervision == 'diar':
            if self.binary_diar_preds:
                diar_preds = (diar_preds > 0.5).float()
            targets = diar_preds
        else:
            raise ValueError(f"Unknown spk_supervision: {self.spk_supervision}, should be 'rttm' or 'diar'")

        (query_pre_encoded_list2, 
        batch_mask
        ) = self.get_query_embs(embs=target_pre_encoded, 
                                embs_len=target_pre_encoded_len, 
                                query_speaker_ids=query_speaker_ids,
                                preds=targets)
        query_pre_encoded1 = query_pre_encoded1[batch_mask]
        query_pre_encoded_len1 = query_pre_encoded_len1[batch_mask]
        transcript = transcript[batch_mask]
        transcript_len = transcript_len[batch_mask]
        target_pre_encoded = target_pre_encoded[batch_mask]
        target_pre_encoded_len = target_pre_encoded_len[batch_mask]
        query_pre_encoded_list1 = [query_pre_encoded1[i, :query_pre_encoded_len1[i]] for i in range(query_pre_encoded1.shape[0])]

        # Step 4: Choose different query embeddings from step 2 and step 3, e.g., query from another query audio or query from the signal itself
        query_pre_encoded, query_pre_encoded_len = self.merge_query_embs(query_pre_encoded_list1, query_pre_encoded_list2)

        query_pre_encoded, query_pre_encoded_len = self.fix_query_length(query_pre_encoded, query_pre_encoded_len)
        
        # Step 5: concatenate query and target embeddings
        pre_encoded, pre_encoded_len = self.concat_query_target(
            query_embs=query_pre_encoded,
            query_lengths=query_pre_encoded_len,
            target_embs=target_pre_encoded,
            target_lengths=target_pre_encoded_len,
        )

        encoded, encoded_len = self.encoder(audio_signal=pre_encoded, length=pre_encoded_len, pre_encode_input=True)

        # Step 4: apply diarization kernel
        if self.diar_kernel_type is not None:
            encoded = self.forward_diar_kernel(encoded, encoded_len, diar_preds)

        return encoded, encoded_len, transcript, transcript_len
    
    def get_query_embs(
        self,
        embs,
        embs_len,
        query_speaker_ids=None,
        preds=None,
    ):
        '''
        Args:
            embs: torch.tensor, shape (B, T, D)
            embs_len: torch.tensor, shape (B)
            preds: torch.tensor, shape (B, T, N)
            targets: torch.tensor, shape (B, T, N)
        
        Returns:
            query_embs: torch.tensor, shape (B, T1, D)
            query_embs_len: torch.tensor, shape (B)
            embs: torch.tensor, shape (B, T, D)
            embs_len: torch.tensor,

        '''
        if not self.overlapping_query:
            non_overlap = (preds.sum(dim=2, keepdim=True) == 1).float()
            preds = preds * non_overlap

        B, T, N = preds.size()
        B, T, D = embs.size()
        preds = torch.stack([preds[i, :, query_speaker_ids[i]] for i in range(B)])

        batch_mask = preds.sum(dim=1) != 0  # Mask where sum is non-zero

        preds, query_speaker_ids = preds[batch_mask], query_speaker_ids[batch_mask]
        embs, embs_len = embs[batch_mask], embs_len[batch_mask] # only select the batch with non-zero sum
        
        query_emb_list = []
        if self.continuous_query: 
            # find the continuous subsequence with the largest mean value
            indices = self.batch_max_avg_sublist_with_limit(mask=preds) # return the start and the end point

            for ib in range(preds.size(0)):
                start, end = indices[ib]
                query_emb = embs[ib, start:end+1]
                query_emb_list.append(query_emb)
            
        else:
            # find the frames with the largest value, but they are not continuous
            indices, _ = self.batch_largest_n_in_order(mask=preds, n=self.max_query_len)
            for ib in range(preds.size(0)):
                query_emb = embs[ib, indices[ib]]
                query_emb_list.append(query_emb)

        return query_emb_list, batch_mask
    
    def fix_query_length(self, query_emb, query_len):
        '''
        Args:
            query_emb: torch.tensor, shape (B, T, D)
            query_len: torch.tensor, shape (B)
        '''
        max_query_len = self.max_query_len
        min_query_len = self.min_query_len
        fixed_query_emb = []
        fixed_query_emb_len = []
        for i in range(query_emb.shape[0]):
            if self.fixed_query_len:
                if query_len[i] < max_query_len:
                    n_repeats = max_query_len // query_len[i] + 1
                    fixed_query_emb.append(query_emb[i].repeat(n_repeats, 1)[:max_query_len])
                else:
                    random_start = torch.randint(0, query_len[i] - max_query_len + 1, (1,)).item()
                    random_end = random_start + max_query_len
                    fixed_query_emb.append(query_emb[i][random_start:random_end])
                fixed_query_emb_len.append(max_query_len)
            else:
                if query_len[i] < max_query_len:
                    n_repeats = max_query_len // query_len[i] + 1
                    padded_query_emb = query_emb[i].repeat(n_repeats, 1)
                else:
                    padded_query_emb = query_emb[i]

                random_len = torch.randint(min_query_len, max_query_len + 1, (1,)).item()
                random_start = torch.randint(0, padded_query_emb.shape[0] - random_len + 1, (1,)).item()
                random_end = random_start + random_len
                fixed_query_emb.append(padded_query_emb[random_start:random_end])
                fixed_query_emb_len.append(random_len)

                # padding zeros to query_embs
                
        if self.fixed_query_len:
            fixed_query_emb = torch.stack(fixed_query_emb)
            fixed_query_emb_len = torch.tensor(fixed_query_emb_len).to(self.device)
        else:
            fixed_query_emb = collate_matrices(fixed_query_emb)
        
        fixed_query_emb_len = torch.tensor(fixed_query_emb_len).to(self.device)
        return fixed_query_emb, fixed_query_emb_len
    
    def merge_query_embs(
        self,
        query_embs1,
        query_embs2,
    ):
        '''
        Args:
            query_embs1: list of torch.tensor, shape [(T1, D)]
            query_embs2: list of torch.tensor, shape [(T2, D)]
        '''
        assert len(query_embs1) == len(query_embs2)

        split_point = len(query_embs1) // 2

        if random.random() < 0.5:
            query_embs = query_embs1[:split_point] + query_embs2[split_point:]
        else:
            query_embs = query_embs2[:split_point] + query_embs1[split_point:]

        query_lengths = [query_emb.size(0) for query_emb in query_embs]
        query_lengths = torch.tensor(query_lengths).to(self.device)

        # padding zeros to query_embs
        query_embs = collate_matrices(query_embs)
        
        return query_embs, query_lengths
    
    
    @torch.no_grad
    def concat_query_target(
        self,
        query_embs,
        query_lengths,
        target_embs,
        target_lengths
    ):
        '''
        Args:
            query_embs: torch.tensor, shape (B, T1, D)
            query_lengths: torch.tensor shape (B)
            target_embs: torch.tensor, shape (B, T2, D)
            target_lengths: torch.tensor, shape (B)
        '''
        zero_embds = torch.zeros_like(target_embs[:, 0:12])
        if self.fixed_query_len:
            query_target_embs = torch.cat([query_embs, zero_embds, target_embs], dim=1)
            query_target_lengths = query_lengths + target_lengths + 12.0
        else:
            query_target_embs = []
            for i in range(query_embs.size(0)):
                query_target_embs.append(torch.cat([query_embs[i][:query_lengths[i]], zero_embds[i], target_embs[i]], dim=0))
            query_target_lengths = query_lengths + target_lengths + 12.0

            # padding zeros to query_target_embs
            query_target_embs = collate_matrices(query_target_embs)
            query_target_lengths = torch.tensor(query_target_lengths).to(self.device)

        return query_target_embs, query_target_lengths, 
    
    def batch_max_avg_sublist_with_limit(self, mask):
        """
        Find the start and end points of the sublist with the largest average value
        with a length constraint in a mask in a batch-wise manner.

        Args:
            mask (torch.Tensor): A tensor of size (B, T) with binary values (0 or 1).
            sub_length (int): Maximum allowed length of the sublist.
        
        Returns:
            torch.Tensor: A tensor of shape (B, 2), where each row contains the start and end indices for the corresponding batch.
        """
        B, L = mask.size()
    
        # Prepare variables to track results
        max_lengths = torch.zeros(B, dtype=torch.long, device=mask.device)
        start_indices = torch.zeros(B, dtype=torch.long, device=mask.device)
        end_indices = torch.zeros(B, dtype=torch.long, device=mask.device)

        # Tracking current streaks
        current_length = torch.zeros(B, dtype=torch.long, device=mask.device)
        current_start = torch.zeros(B, dtype=torch.long, device=mask.device)

        for l in range(L):
            # Update current streak length
            active = mask[:, l] == 1
            not_active = mask[:, l] == 0

            # Update current lengths and reset where not active
            current_length[active] += 1
            current_length[not_active] = 0

            # Set current_start where streak begins
            current_start[active & (current_length == 1)] = l

            # Update max_lengths, start_indices, and end_indices
            update_mask = current_length > max_lengths
            max_lengths[update_mask] = current_length[update_mask]
            start_indices[update_mask] = current_start[update_mask]
            end_indices[update_mask] = l  # Update end index to the current position

        return torch.stack([start_indices, end_indices], dim=1)
    
    def batch_largest_n_in_order(self, mask, n, threshold=0.5):
        B, T = mask.size()

        # Mask out values smaller than the threshold
        mask = torch.where(mask >= threshold, mask, torch.tensor(float('-inf'), device=mask.device))

        # Initialize the result list
        indices, values = [], []

        # Populate the result list with indices and values while maintaining the original order and removing -inf
        for i in range(B):
            valid_indices = torch.arange(T, device=mask.device)[mask[i] >= threshold]
            valid_values = mask[i][mask[i] >= threshold]
            # Retain the original order of valid values
            if len(valid_values) > 0:
                top_indices = valid_indices[:n]
                top_values = valid_values[:n]
                indices.append(top_indices)
                values.append(top_values)
            else:
                indices.append(torch.tensor([], device=mask.device, dtype=torch.long))
                values.append(torch.tensor([], device=mask.device, dtype=mask.dtype))

        return indices, values 

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextSpkBpeDataset(cfg = config, tokenizer=self.tokenizer,),
            )
        
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, query, query_len, transcript, transcript_len, spk_targets, query_spk_ids = batch

        encoded, encoded_len, transcript, transcript_len = self.forward_train_val(
            signal=signal, signal_len=signal_len, query=query, query_len=query_len, transcript=transcript, transcript_len=transcript_len, spk_targets=spk_targets, query_speaker_ids=query_spk_ids
        )
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}
    
    @torch.no_grad
    def forward_with_gt_query(
        self, 
        signal=None,
        signal_len=None,
        query=None, 
        query_len=None,
    ):
        query_pre_encoded, query_pre_encoded_len = self.forward_pre_encode(input_signal=query, input_signal_length=query_len)
        target_pre_encoded, target_pre_encoded_len = self.forward_pre_encode(input_signal=signal, input_signal_length=signal_len)
        
        query_pre_encoded, query_pre_encoded_len = self.fix_query_length(query_pre_encoded, query_pre_encoded_len)
        
        target_pre_encoded = target_pre_encoded.repeat(query_pre_encoded.shape[0], 1, 1)
        target_pre_encoded_len = target_pre_encoded_len.repeat(query_pre_encoded.shape[0])

        query_target_encoded, query_target_encoded_len = self.concat_query_target(
            query_embs=query_pre_encoded,
            query_lengths=query_pre_encoded_len,
            target_embs=target_pre_encoded,
            target_lengths=target_pre_encoded_len,
        )

        encoded, encoded_len = self.encoder(audio_signal=query_target_encoded, length=query_target_encoded_len, pre_encode_input=True)

        return encoded, encoded_len
    
    @torch.no_grad
    def forward_with_pred_query(
        self, 
        signal=None,
        signal_len=None,
    ):
        target_pre_encoded, target_pre_encoded_len = self.forward_pre_encode(input_signal=signal, input_signal_length=signal_len)
        diar_preds, diar_pre_encoded, diar_pre_encoded_len = self.forward_diar(audio_signal=signal, audio_signal_length=signal_len)

        # Step 2: find the query embeddings from the pre-encoded embeddings based on diar predictions     
        # diar_preds: B x T x N
        diar_preds = (diar_preds > 0.5).float()
        n_spk = diar_preds.size(2)
        query_speaker_ids = torch.arange(n_spk).to(self.device)
        target_pre_encoded = target_pre_encoded.repeat(n_spk, 1, 1)
        target_pre_encoded_len = target_pre_encoded_len.repeat(n_spk)
        diar_preds = diar_preds.repeat(n_spk, 1, 1)

        (query_pre_encoded_list, 
        batch_mask
        ) = self.get_query_embs(embs=target_pre_encoded, 
                                embs_len=target_pre_encoded_len, 
                                query_speaker_ids=query_speaker_ids,
                                preds=diar_preds)

        query_pre_encoded_len = [query_pre_encoded.size(0) for query_pre_encoded in query_pre_encoded_list]
        query_pre_encoded_len = torch.tensor(query_pre_encoded_len).to(self.device)
        query_pre_encoded = collate_matrices(query_pre_encoded_list)

        target_pre_encoded = target_pre_encoded[batch_mask]
        target_pre_encoded_len = target_pre_encoded_len[batch_mask]

        query_pre_encoded, query_pre_encoded_len = self.fix_query_length(query_pre_encoded, query_pre_encoded_len)

        # Step 3: concatenate query and target embeddings
        pre_encoded, pre_encoded_len = self.concat_query_target(
            query_embs=query_pre_encoded,
            query_lengths=query_pre_encoded_len,
            target_embs=target_pre_encoded,
            target_lengths=target_pre_encoded_len,
        )

        encoded, encoded_len = self.encoder(audio_signal=pre_encoded, length=pre_encoded_len, pre_encode_input=True)

        return encoded, encoded_len
    
    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, query, query_len, transcript, transcript_len, spk_targets, query_spk_ids = batch

        encoded, encoded_len, transcript, transcript_len = self.forward_train_val(
            signal=signal, signal_len=signal_len, query=query, query_len=query_len, transcript=transcript, transcript_len=transcript_len, spk_targets=spk_targets, query_speaker_ids=query_spk_ids
        )
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs
    
    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        signal, signal_len, query, query_len, spk_targets = batch


        # encoded, encoded_len = self.forward_with_pred_query(
        #     signal=signal, signal_len=signal_len
        # )
        encoded, encoded_len = self.forward_with_gt_query(
            signal=signal, signal_len=signal_len, query=query, query_len=query_len
        )
        del signal
        output = dict(encoded=encoded, encoded_len=encoded_len, spk_targets=spk_targets)

        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Tuple[List['Hypothesis'], List['Hypothesis']]:
        encoded = outputs.pop('encoded')
        encoded_len = outputs.pop('encoded_len')
        spk_targets = outputs.pop('spk_targets')

        best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=trcfg.return_hypotheses,
            partial_hypotheses=trcfg.partial_hypothesis,
        )

        # cleanup memory
        del encoded, encoded_len

        hypotheses = []
        all_hypotheses = []

        hypotheses += best_hyp
        if all_hyp is not None:
            all_hypotheses += all_hyp
        else:
            all_hypotheses += best_hyp
        self.total_results.append(hypotheses)
        # self.preds_rttms.append(spk_targets)

        return (hypotheses, all_hypotheses)

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'dataset_manifest' in config:
            manifest_filepath = config['dataset_manifest']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': True,
            'use_bucketing': False,
            'channel_selector': config.get('channel_selector', None),
            'inference_mode': self.cfg.test_ds.get('inference_mode', True)
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))

        return temporary_datalayer
    
    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray/tensor audio array or path to a manifest file.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            partial_hypothesis: Optional[List['Hypothesis']] - A list of partial hypotheses to be used during rnnt
                decoding. This is useful for streaming rnnt decoding. If this is not None, then the length of this
                list should be equal to the length of the audio list.
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        self.total_results = []
        self.preds_rttms = []
        super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            override_config=override_config,
            # Additional arguments
            partial_hypothesis=partial_hypothesis,
        )

        return self.total_results
