# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Any, Dict, List, Optional

import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_lhotse_target_speaker import LhotseSpeechToTextTgtSpkBpeDataset
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins import TranscribeConfig
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging


class EncDecHybridRNNTCTCTgtSpkBPEModel(EncDecHybridRNNTCTCBPEModel):
    """Encoder decoder Hybrid RNNT-CTC model for target speaker ASR."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        if 'diar_model_path' in self.cfg:
            pass
        else:
            raise ValueError("`diar_model_path` is required for target speaker ASR model.")
        # Initialize the speaker branch
        self.diarization_model = None
        self._init_diar_model()

        self.num_speakers = cfg.model_defaults.get('num_speakers', 4)

        # layer normalization, ln, l2, or None
        self.norm = cfg.get('norm', None)

        if cfg.norm == 'ln':
            self.asr_norm = torch.nn.LayerNorm(cfg.model_defaults.enc_hidden)
            self.diar_norm = torch.nn.LayerNorm(4)

        self.kernel_norm = cfg.get('kernel_norm', None)

        # projection layer
        self.diar_kernel_type = cfg.get('diar_kernel_type', None)

        proj_in_size = self.num_speakers + cfg.model_defaults.enc_hidden
        proj_out_size = cfg.model_defaults.enc_hidden
        self.joint_proj = torch.nn.Sequential(
            torch.nn.Linear(proj_in_size, proj_out_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_out_size * 2, proj_out_size),
        )
        self.diar_kernal = self.joint_proj

    def _init_diar_model(self) -> None:
        """Initialize the speaker model."""
        model_path = self.cfg.diar_model_path
        pretrained_diar_model = None
        if model_path.endswith('.nemo'):
            pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
            logging.info(f"Diarization Model restored locally from {model_path}")
        elif model_path.endswith('.ckpt'):
            pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info(f"Diarization Model restored locally from {model_path}")
        else:
            logging.info("Model path incorrect")

        self.diarization_model = pretrained_diar_model

        # diarization model streaming mode is not supported yet in TS-ASR model
        self.diarization_model.streaming_mode = False

        if self.cfg.freeze_diar:
            self.diarization_model.eval()

    def forward_diar(
        self, input_signal: Optional[torch.Tensor] = None, input_signal_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for diarization model."""
        preds = self.diarization_model.forward(audio_signal=input_signal, audio_signal_length=input_signal_length)
        return preds

    def fix_diar_output(self, diar_pred: torch.Tensor, asr_frame_count: int) -> torch.Tensor:
        """
        Duct-tape solution for extending the speaker predictions to the length of the ASR output
        or truncate the speaker predictions to the length of the ASR output.
        """
        if diar_pred.shape[1] < asr_frame_count:
            last_emb = diar_pred[:, -1, :].unsqueeze(1)
            additional_frames = asr_frame_count - diar_pred.shape[1]
            last_repeats = last_emb.repeat(1, additional_frames, 1)
            extended_diar_preds = torch.cat((diar_pred, last_repeats), dim=1)
            return extended_diar_preds

        # temporary solution if diar_pred longer than encoded
        return diar_pred[:, :asr_frame_count, :]

    def _get_probablistic_mix(
        self, diar_preds: torch.Tensor, spk_targets: torch.Tensor, rttm_mix_prob: float = 0.0
    ) -> torch.Tensor:
        """
        Sample a probablistic mixture of speaker labels for each time step then apply it to
        the diarization predictions and the speaker targets.

        Args:
            diar_preds (Tensor): Tensor of shape [B, T, D] representing the diarization predictions.
            spk_targets (Tensor): Tensor of shape [B, T, D] representing the speaker targets.
            rttm_mix_prob (float): Probability of mixing RTTM targets.

        Returns:
            torch.Tensor: Tensor of shape [B, T, D] representing the
            probablistic mixture of speaker labels for each time step.
        """
        batch_probs_raw = torch.distributions.categorical.Categorical(
            probs=torch.tensor([(1 - rttm_mix_prob), rttm_mix_prob]).repeat(diar_preds.shape[0], 1)
        ).sample()
        batch_probs = (
            batch_probs_raw.view(diar_preds.shape[0], 1, 1).repeat(1, diar_preds.shape[1], diar_preds.shape[2])
        ).to(diar_preds.device)
        batch_diar_preds = (1 - batch_probs) * diar_preds + batch_probs * spk_targets
        return batch_diar_preds

    def forward(self, input_signal: torch.Tensor, input_signal_length: torch.Tensor, spk_targets: torch.Tensor):
        """
        Forward pass for training and validation of getting encoder outputs.

        Please refer to training_step and validation_pass for full model forward pass.

        Args:
            input_signal (torch.Tensor): The input audio signal of shape [B, T, C].
            input_signal_length (torch.Tensor): The length of the input audio signal of shape [B].
            spk_targets (torch.Tensor): The speaker targets of shape [B, T, D].

        Returns:
            tuple: (encoded, encoded_len) where encoded is the encoded features and encoded_len is the lengths.
        """
        signal = input_signal
        signal_len = input_signal_length
        diar_preds = None
        if self.cfg.spk_supervision_strategy == 'rttm':
            if spk_targets is not None:
                diar_preds = spk_targets
            else:
                raise ValueError("`spk_targets` is required for speaker supervision strategy 'rttm'")
        elif self.cfg.spk_supervision_strategy == 'diar':
            with torch.set_grad_enabled(not self.cfg.freeze_diar):
                diar_preds = self.forward_diar(signal, signal_len)
            if diar_preds is None:
                raise ValueError("`diar_pred`is required for speaker supervision strategy 'diar'")
        elif self.cfg.spk_supervision_strategy == 'mix':
            with torch.set_grad_enabled(not self.cfg.freeze_diar):
                diar_preds = self.forward_diar(signal, signal_len)
            # resolve mismatch between spk_targets and diar_preds
            spk_targets = self.fix_diar_output(spk_targets, diar_preds.shape[1])
            diar_preds = self._get_probablistic_mix(
                diar_preds=diar_preds, spk_targets=spk_targets, rttm_mix_prob=float(self.cfg.rttm_mix_prob)
            )
        else:
            raise ValueError(f"Invalid RTTM strategy {self.cfg.spk_supervision_strategy} is not supported.")

        encoded, encoded_len = super().forward(input_signal=signal, input_signal_length=signal_len)

        encoded = torch.transpose(encoded, 1, 2)  # B * D * T -> B * T * D
        if diar_preds.shape[1] != encoded.shape[1]:
            # KD duct-tape solution for extending the speaker predictions
            asr_frame_count = encoded.shape[1]
            diar_preds = self.fix_diar_output(diar_preds, asr_frame_count)
        # Normalize the features
        if self.norm == 'ln':
            diar_preds = self.diar_norm(diar_preds)
            encoded = self.asr_norm(encoded)
        elif self.norm == 'l2':
            diar_preds = torch.nn.functional.normalize(diar_preds, p=2, dim=-1)
            encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)

        if diar_preds.shape[1] > encoded.shape[1]:
            diar_preds = diar_preds[:, : encoded.shape[1], :]
        if self.diar_kernel_type == 'projection':
            concat_enc_states = torch.cat([encoded, diar_preds], dim=-1)
            encoded = self.joint_proj(concat_enc_states)
        else:
            raise ValueError(
                f"Invalid diar_kernel_type {self.diar_kernel_type} is not supported. "
                "Choose from 'projection'. Others will be supported soon"
            )

        del signal
        encoded = torch.transpose(encoded, 1, 2)  # B * T * D -> B * D * T
        return encoded, encoded_len

    # training_step with new forward
    def training_step(self, batch, batch_nb):
        """Training step with target speaker support."""
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len, spk_targets = batch
        encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len, spk_targets=spk_targets
        )

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        compute_wer = (sample_id + 1) % log_every_n_steps == 0

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

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
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

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        """Validation pass with target speaker support."""
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len, spk_targets = batch
        encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len, spk_targets=spk_targets
        )

        tensorboard_logs = {}
        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)
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
                decoded, target_len, _ = self.decoder(targets=transcript, target_length=transcript_len)
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

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value
        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        """Setup dataloader from config."""
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextTgtSpkBpeDataset(
                    cfg=config,
                    tokenizer=self.tokenizer,
                ),
                tokenizer=self.tokenizer,
            )

        return None

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
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,  # pylint: disable=protected-access
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': True,
            'use_bucketing': False,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
            'num_speakers': self.cfg.test_ds.get('num_speakers', 4),
            'spk_tar_all_zero': self.cfg.test_ds.get('spk_tar_all_zero', False),
            'num_sample_per_mel_frame': self.cfg.test_ds.get('num_sample_per_mel_frame', 160),
            'num_mel_frame_per_asr_frame': self.cfg.test_ds.get('num_mel_frame_per_asr_frame', 8),
            'separater_freq': self.cfg.test_ds.get('separater_freq', 500),
            'separater_duration': self.cfg.test_ds.get('separater_duration', 1),
            'separater_unvoice_ratio': self.cfg.test_ds.get('separater_unvoice_ratio', 0.3),
            'add_special_token': self.cfg.test_ds.get('add_special_token', True),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step with target speaker support."""
        signal, signal_len, transcript, _, spk_targets = batch

        encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len, spk_targets=spk_targets
        )
        del signal

        if self.cur_decoder == 'rnnt':
            best_hyp = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
            )
        else:
            logits = self.ctc_decoder(encoder_output=encoded)
            best_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=logits,
                decoder_lengths=encoded_len,
                return_hypotheses=True,
            )

        return list(zip(transcript, best_hyp))

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):  # pylint: disable=unused-argument
        """Forward pass for transcription with target speaker support."""
        signal, signal_len, _, _, spk_targets = batch

        encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len, spk_targets=spk_targets
        )

        if self.cur_decoder == 'rnnt':
            output = {"encoded": encoded, "encoded_len": encoded_len}
        elif self.cur_decoder == 'ctc':
            logits = self.ctc_decoder(encoder_output=encoded)
            output = {"logits": logits, "encoded_len": encoded_len}
        else:
            raise ValueError(f"Invalid decoder type: {self.cur_decoder}")

        del encoded
        return output
