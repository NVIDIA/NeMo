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
# 
import copy
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

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

class EncDecRNNTBPEMDTSASRModel(EncDecRNNTBPEModel):
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

        self.num_speakers = cfg.model_defaults.get('num_speakers', 4)
        if 'diar_model_path' in self.cfg:
            self._init_diar_model()
            
            self.diar_kernel_type = cfg.get('diar_kernel', None)
            self.binary_diar_preds = cfg.get('binary_diar_preds', True)
            self.spk_supervision = cfg.get('spk_supervision', 'rttm')

            if self.diar_kernel_type == 'metacat':
                # projection layer
                proj_in_size = cfg.model_defaults.enc_hidden
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
        
        # self.decoders = [self.decoder]
        # self.joints = [self.joint]

        # for i in range(1, self.num_speakers):
        #     self.decoders.append(copy.deepcopy(self.decoder))
        #     self.joints.append(copy.deepcopy(self.joint))

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
        """
        Args:
            encoded: torch.tensor, shape (B, D, T)
            encoded_len: torch.tensor, shape (B)
            diar_preds: torch.tensor, shape (B, T, N)
        """
        encoded_list = []
        if self.diar_kernel_type == 'metacat_residule':
            concat_enc_states_list = []
            BS = encoded.size(0)
            n_spk = diar_preds.size(2)
            for i_spk in range(diar_preds.size(2)):
                concat_enc_states = encoded * diar_preds[:, :, i_spk].unsqueeze(1)
                concat_enc_states_list.append(concat_enc_states)
            concat_enc_states = torch.cat(concat_enc_states_list, dim=0)
            all_encoded = self.joint_proj(concat_enc_states.transpose(1, 2)).transpose(1, 2)
            encoded_list = [encoded + all_encoded[i*BS:(i+1)*BS] for i in range(n_spk)]
        elif self.diar_kernel_type == 'metacat':
            concat_enc_states_list = []
            BS = encoded.size(0)
            n_spk = diar_preds.size(2)
            for i_spk in range(diar_preds.size(2)):
                concat_enc_states = encoded * diar_preds[:, :, i_spk].unsqueeze(1)
                concat_enc_states_list.append(concat_enc_states)
            concat_enc_states = torch.cat(concat_enc_states_list, dim=0)
            all_encoded = self.joint_proj(concat_enc_states.transpose(1, 2)).transpose(1, 2)
            encoded_list = [all_encoded[i*BS:(i+1)*BS] for i in range(n_spk)]

        return encoded_list

    def forward_train_val(
        self, 
        signal=None,
        signal_len=None,
        spk_targets=None,
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
        processed_signal, processed_signal_length = self.preprocessor(
                input_signal=signal,
                length=signal_len,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length, pre_encode_input=False)

        # Step 4: apply diarization kernel
        if self.diar_kernel_type:
            encoded_list = self.forward_diar_kernel(encoded, encoded_len, spk_targets)
            encoded_len_list = [encoded_len for _ in range(len(encoded_list))]

            return encoded_list, encoded_len_list
        
        return encoded, encoded_len

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextMSpkBpeDataset(cfg = config, tokenizer=self.tokenizer,),
            )
        
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript_list, transcript_len_list, spk_targets = batch

        encoded_list, encoded_len_list = self.forward_train_val(
            signal=signal, signal_len=signal_len, spk_targets=spk_targets
        )
        del signal

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        losses, wers = [], []
        for i in range(0, 1):
            # self.joints[i].to(self.device)
            # self.decoders[i].to(self.device)

            if isinstance(encoded_list, list):
                encoded, encoded_len = encoded_list[i], encoded_len_list[i]
            else:
                encoded, encoded_len = encoded_list, encoded_len_list
            transcript, transcript_len = transcript_list[i], transcript_len_list[i]

            # During training, loss must be computed, so decoder forward is necessary
            decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)


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
                losses.append(loss_value)
                wers.append(wer)

                # Reset access registry
                if AccessMixin.is_access_enabled(self.model_guid):
                    AccessMixin.reset_registry(self)

            # break # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO, only one speaker for now

        loss_value = losses[0]
        
        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        if compute_wer:
            wer = sum(wers) / len(wers) if len(wers) > 0 else None
            tensorboard_logs.update({'training_batch_wer': wer})
            for i, wer in enumerate(wers):
                tensorboard_logs.update({f'training_batch_wer_spk_{i}': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    
    def validation_pass(self, batch, batch_idx, dataloader_idx=0):

        signal, signal_len, transcript_list, transcript_len_list, spk_targets = batch

        encoded_list, encoded_len_list = self.forward_train_val(
            signal=signal, signal_len=signal_len, spk_targets=spk_targets
        )
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        losses, wers = [], []
        wer_nums, wer_denoms = [], []
        for i in range(0, 1):
            if isinstance(encoded_list, list):
                encoded, encoded_len = encoded_list[i], encoded_len_list[i]
            else:
                encoded, encoded_len = encoded_list, encoded_len_list
            transcript, transcript_len = transcript_list[i], transcript_len_list[i]

            # self.joints[i].to(self.device)
            # self.decoders[i].to(self.device)
            
            tensorboard_logs = {}

            # If experimental fused Joint-Loss-WER is not used
            if not self.joint.fuse_loss_wer:
                if self.compute_eval_loss:
                    decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                    joint = self.joint[i](encoder_outputs=encoded, decoder_outputs=decoder)

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

                losses.append(loss_value)
                wers.append(wer)
                wer_nums.append(wer_num)
                wer_denoms.append(wer_denom)

        if loss_value is not None:
            loss_value = sum(losses) / len(losses) if len(losses) > 0 else None
            tensorboard_logs['val_loss'] = loss_value

        tensorboard_logs['val_wer_num'] = sum(wer_nums) / len(wer_nums) if len(wer_nums) > 0 else None
        tensorboard_logs['val_wer_denom'] = sum(wer_denoms) / len(wer_denoms) if len(wer_denoms) > 0 else None
        tensorboard_logs['val_wer'] = sum(wers) / len(wers) if len(wers) > 0 else None

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs
    
    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        # signal, signal_len, query, query_len, transcript, transcript_len, spk_targets = batch

        # encoded, encoded_len, transcript, transcript_len = self.forward_with_pred_query(
        #     signal=signal, signal_len=signal_len, transcript=transcript, transcript_len=transcript_len
        # )
        # if encoded.shape[0] == 1:
        #     encoded, encoded_len, transcript, transcript_len = self.forward_with_gt_query(
        #         signal=signal, signal_len=signal_len, query=query, query_len=query_len, transcript=transcript, transcript_len=transcript_len
        #     )
        signal, signal_len, transcript_list, transcript_len_list, spk_targets = batch

        encoded, encoded_len = self.forward_train_val(
            signal=signal, signal_len=signal_len, spk_targets=spk_targets
        )
        del signal
        output = dict(encoded=encoded[1], encoded_len=encoded_len[1], spk_targets=spk_targets)

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