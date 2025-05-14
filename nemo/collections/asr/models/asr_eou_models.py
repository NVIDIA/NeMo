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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_eou_label_lhotse import (
    EOB_LABEL,
    EOB_STRING,
    EOU_LABEL,
    EOU_STRING,
    AudioToTextEOUBatch,
    LhotseSpeechToTextBpeEOUDataset,
)
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.eou_utils import (
    EOUResult,
    cal_eou_metrics_from_frame_labels,
    flatten_nested_list,
)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.utils import move_data_to_device
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging

__all__ = ['EncDecRNNTBPEEOUModel']


@dataclass
class EOUPrediction:
    eou_probs: Optional[List[float]] = None
    eob_probs: Optional[List[float]] = None
    eou_preds: Optional[List[bool]] = None
    eob_preds: Optional[List[bool]] = None


class EncDecRNNTBPEEOUModel(EncDecRNNTBPEModel):
    def __init__(self, cfg: DictConfig, trainer):

        self._patch_decoding_cfg(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.eou_token = self.tokenizer.token_to_id(EOU_STRING)
        self.eob_token = self.tokenizer.token_to_id(EOB_STRING)
        self.frame_len_in_secs = self.cfg.preprocessor.window_stride * self.cfg.encoder.subsampling_factor

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
            return_hypotheses=True,
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

    def _patch_decoding_cfg(self, cfg: DictConfig):
        """
        Patch the decoding config as needed for EOU computation
        """
        with open_dict(cfg):
            if cfg.decoding.strategy in ['greedy', 'greedy_batch']:
                cfg.decoding.greedy.preserve_alignments = True
                cfg.decoding.greedy.compute_timestamps = True
            elif cfg.decoding.strategy in ['beam', 'tsd', 'alsd', 'maes']:
                cfg.decoding.beam.preserve_alignments = True
                cfg.decoding.beam.compute_timestamps = True

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """
        PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
        """
        batch = move_data_to_device(batch, device)
        return batch

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        cfg = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        dataset = LhotseSpeechToTextBpeEOUDataset(
            cfg=cfg, tokenizer=self.tokenizer, return_cuts=config.get("do_transcribe", False)
        )
        return get_lhotse_dataloader_from_config(
            config,
            # During transcription, the model is initially loaded on the CPU.
            # To ensure the correct global_rank and world_size are set,
            # these values must be passed from the configuration.
            global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
            world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
            dataset=dataset,
            tokenizer=self.tokenizer,
        )

    def training_step(self, batch: AudioToTextEOUBatch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
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

    def predict_step(self, batch: AudioToTextEOUBatch, batch_idx, dataloader_idx=0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        return list(best_hyp_text)

    def validation_pass(self, batch: AudioToTextEOUBatch, batch_idx: int, dataloader_idx: int = 0):
        signal = batch.audio_signal
        signal_len = batch.audio_lengths
        transcript = batch.text_tokens
        transcript_len = batch.text_token_lengths

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}
        text_gt = self._get_text_from_tokens(transcript, transcript_len)
        tensorboard_logs['val_sample_id'] = batch.sample_ids
        tensorboard_logs['val_audio_filepath'] = batch.audio_filepaths
        tensorboard_logs['val_text_gt'] = text_gt
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
            hypotheses = self.wer.get_hypotheses()

            text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])
            eou_predictions = self.get_eou_predictions_from_hypotheses(hypotheses, batch)
            eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)

            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer
            tensorboard_logs['val_eou_metrics'] = eou_metrics_list
            tensorboard_logs['val_eob_metrics'] = eob_metrics_list
            tensorboard_logs['val_text_pred'] = text_pred

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
                keep_hypotheses=True,
            )

            hypotheses = self.joint.get_hypotheses()
            text_pred = self._get_text_from_tokens([x.y_sequence for x in hypotheses])

            eou_predictions = self.get_eou_predictions_from_hypotheses(hypotheses, batch)

            eou_metrics_list, eob_metrics_list = self._calculate_eou_metrics(eou_predictions, batch)

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer
            tensorboard_logs['val_eou_metrics'] = eou_metrics_list
            tensorboard_logs['val_eob_metrics'] = eob_metrics_list
            tensorboard_logs['val_text_pred'] = text_pred

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def _get_text_from_tokens(self, tokens: torch.Tensor, tokens_len: Optional[torch.Tensor] = None) -> List[str]:
        """
        Convert tokens to text.
        Args:
            tokens: tensor of tokens
        Returns:
            text: list of text
        """
        text_list = []
        for i in range(len(tokens)):
            tokens_i = tokens[i]
            if tokens_len is not None:
                tokens_i = tokens[i][: tokens_len[i]]
            text = self.tokenizer.ids_to_text(tokens_i)
            text_list.append(text)
        return text_list

    def get_eou_predictions_from_hypotheses(
        self, hypotheses: List[Hypothesis], batch: AudioToTextEOUBatch
    ) -> List[EOUPrediction]:
        """
        Get EOU predictions from the hypotheses.
        Args:
            hypotheses: batch of hypotheses
        Returns:
            eou_predictions: list of EOU predictions
        """
        eou_predictions = []

        for hyp in hypotheses:
            # Process one hypothesis at a time
            eou_probs = []
            eob_probs = []
            eou_preds = []
            eob_preds = []
            for alignment in hyp.alignments:
                # Process for each timestamp
                probs = torch.softmax(torch.stack([a[0] for a in alignment], dim=0), dim=-1)  # unfold RNNT preds
                tokens = torch.stack([a[1] for a in alignment], dim=0)  # unfold RNNT preds
                # Get the max prob for eou and eob
                # and check if eou and eob are predicted
                max_eou_prob = probs[:, self.eou_token].max().item()
                max_eob_prob = probs[:, self.eob_token].max().item()
                eou_pred = torch.any(tokens == self.eou_token).item()
                eob_pred = torch.any(tokens == self.eob_token).item()

                eou_probs.append(max_eou_prob)
                eob_probs.append(max_eob_prob)
                eou_preds.append(eou_pred)
                eob_preds.append(eob_pred)

            eou_predictions.append(
                EOUPrediction(
                    eou_probs=eou_probs,
                    eob_probs=eob_probs,
                    eou_preds=eou_preds,
                    eob_preds=eob_preds,
                )
            )

        return eou_predictions

    def _pad_to_same_length(self, eou_labels: List[float], eou_preds: List[float]) -> Tuple[List[float], List[float]]:
        """
        Pad the EOU labels and predictions to the same length.
        Args:
            eou_labels: list of EOU labels
            eou_preds: list of EOU predictions
        Returns:
            eou_labels: list of EOU labels, padded to the same length
            eou_preds: list of EOU predictions, padded to the same length
        """
        if len(eou_labels) < len(eou_preds):
            eou_labels = eou_labels + [0] * (len(eou_preds) - len(eou_labels))
        elif len(eou_labels) > len(eou_preds):
            eou_preds = eou_preds + [0] * (len(eou_labels) - len(eou_preds))
        return eou_labels, eou_preds

    def _calculate_eou_metrics(
        self, eou_predictions: List[EOUPrediction], batch: AudioToTextEOUBatch
    ) -> Tuple[List, List]:
        """
        Calculate EOU metrics.
        Args:
            eou_predictions: list of EOU predictions
            batch: batch of data
        Returns:
            eou_metrics_list: list of EOU metrics, each is of type EOUResult
            eob_metrics_list: list of EOB metrics, each is of type EOUResult
        """
        # Get the ground truth EOU labels
        eou_labels = batch.eou_targets
        eou_labels_len = batch.eou_target_lengths

        # Calculate EOU metrics
        eou_metrics_list = []
        eob_metrics_list = []
        for i, eou_prediction in enumerate(eou_predictions):
            eou_preds_i = [float(x) for x in eou_prediction.eou_preds]
            eob_preds_i = [float(x) for x in eou_prediction.eob_preds]

            eou_labels_i = (eou_labels[i][: eou_labels_len[i]] == EOU_LABEL).float().tolist()
            eob_labels_i = (eou_labels[i][: eou_labels_len[i]] == EOB_LABEL).float().tolist()

            # Pad the EOU labels and predictions to the same length with zeros
            eou_labels_i, eou_preds_i = self._pad_to_same_length(eou_labels_i, eou_preds_i)
            eob_labels_i, eob_preds_i = self._pad_to_same_length(eob_labels_i, eob_preds_i)

            # Calculate EOU metrics
            eou_metrics = cal_eou_metrics_from_frame_labels(
                prediction=eou_preds_i,
                reference=eou_labels_i,
                threshold=0.0,
                collar=0.0,
                frame_len_in_secs=self.frame_len_in_secs,
            )  # type: EOUResult

            eob_metrics = cal_eou_metrics_from_frame_labels(
                prediction=eob_preds_i,
                reference=eob_labels_i,
                threshold=0.0,
                collar=0.0,
                frame_len_in_secs=self.frame_len_in_secs,
            )

            eou_metrics_list.append(eou_metrics)
            eob_metrics_list.append(eob_metrics)

        return eou_metrics_list, eob_metrics_list

    def multi_inference_epoch_end(self, outputs, dataloader_idx: int = 0, mode: str = "val"):
        assert mode in ['val', 'test'], f"Invalid mode: {mode}. Must be 'val' or 'test'."
        self._maybe_save_predictions(outputs, mode=mode, dataloader_idx=dataloader_idx)

        # Aggregate WER metrics
        if self.compute_eval_loss:
            loss_mean = torch.stack([x[f'{mode}_loss'] for x in outputs]).mean()
            loss_log = {f'{mode}_loss': loss_mean}
        else:
            loss_log = {}
        wer_num = torch.stack([x[f'{mode}_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x[f'{mode}_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**loss_log, f'{mode}_wer': wer_num.float() / wer_denom}

        # Aggregate EOU/EOB metrics
        eou_metrics = []  # type: List[EOUResult]
        eob_metrics = []  # type: List[EOUResult]
        for x in outputs:
            eou_metrics.extend(x[f'{mode}_eou_metrics'])
            eob_metrics.extend(x[f'{mode}_eob_metrics'])

        num_eou_utterances = sum([x.num_utterances for x in eou_metrics])
        eou_latency = flatten_nested_list([x.latency for x in eou_metrics])
        eou_early_cutoff = flatten_nested_list([x.early_cutoff for x in eou_metrics])

        num_eob_utterances = sum([x.num_utterances for x in eob_metrics])
        eob_latency = flatten_nested_list([x.latency for x in eob_metrics])
        eob_early_cutoff = flatten_nested_list([x.early_cutoff for x in eob_metrics])

        eou_avg_num_early_cutoff = len(eou_early_cutoff) / num_eou_utterances
        eob_avg_num_early_cutoff = len(eob_early_cutoff) / num_eob_utterances
        if len(eou_latency) == 0:
            eou_latency = [0.0]
        if len(eou_early_cutoff) == 0:
            eou_early_cutoff = [0.0]
        if len(eob_latency) == 0:
            eob_latency = [0.0]
        if len(eob_early_cutoff) == 0:
            eob_early_cutoff = [0.0]

        eou_missing = [x.missing for x in eou_metrics]
        eob_missing = [x.missing for x in eob_metrics]

        eou_latency = torch.tensor(eou_latency)
        eou_latency_p90 = torch.quantile(eou_latency, 0.9).item()
        eou_latency_p95 = torch.quantile(eou_latency, 0.95).item()
        eou_latency_p99 = torch.quantile(eou_latency, 0.99).item()

        eou_early_cutoff = torch.tensor(eou_early_cutoff)
        eou_early_cutoff_p90 = torch.quantile(eou_early_cutoff, 0.9).item()
        eou_early_cutoff_p95 = torch.quantile(eou_early_cutoff, 0.95).item()
        eou_early_cutoff_p99 = torch.quantile(eou_early_cutoff, 0.99).item()

        eob_latency = torch.tensor(eob_latency)
        eob_latency_p90 = torch.quantile(eob_latency, 0.9).item()
        eob_latency_p95 = torch.quantile(eob_latency, 0.95).item()
        eob_latency_p99 = torch.quantile(eob_latency, 0.99).item()

        eob_early_cutoff = torch.tensor(eob_early_cutoff)
        eob_early_cutoff_p90 = torch.quantile(eob_early_cutoff, 0.9).item()
        eob_early_cutoff_p95 = torch.quantile(eob_early_cutoff, 0.95).item()
        eob_early_cutoff_p99 = torch.quantile(eob_early_cutoff, 0.99).item()

        tensorboard_logs[f'{mode}_eou_latency_p90'] = eou_latency_p90
        tensorboard_logs[f'{mode}_eou_latency_p95'] = eou_latency_p95
        tensorboard_logs[f'{mode}_eou_latency_p99'] = eou_latency_p99

        tensorboard_logs[f'{mode}_eou_early_cutoff_p90'] = eou_early_cutoff_p90
        tensorboard_logs[f'{mode}_eou_early_cutoff_p95'] = eou_early_cutoff_p95
        tensorboard_logs[f'{mode}_eou_early_cutoff_p99'] = eou_early_cutoff_p99

        tensorboard_logs[f'{mode}_eob_latency_p90'] = eob_latency_p90
        tensorboard_logs[f'{mode}_eob_latency_p95'] = eob_latency_p95
        tensorboard_logs[f'{mode}_eob_latency_p99'] = eob_latency_p99

        tensorboard_logs[f'{mode}_eob_early_cutoff_p90'] = eob_early_cutoff_p90
        tensorboard_logs[f'{mode}_eob_early_cutoff_p95'] = eob_early_cutoff_p95
        tensorboard_logs[f'{mode}_eob_early_cutoff_p99'] = eob_early_cutoff_p99

        tensorboard_logs[f'{mode}_eou_early_cutoff_avg_num'] = eou_avg_num_early_cutoff
        tensorboard_logs[f'{mode}_eob_early_cutoff_avg_num'] = eob_avg_num_early_cutoff

        tensorboard_logs[f'{mode}_eou_missing'] = sum(eou_missing) / num_eou_utterances
        tensorboard_logs[f'{mode}_eob_missing'] = sum(eob_missing) / num_eob_utterances

        return {**loss_log, 'log': tensorboard_logs}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='val')

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_inference_epoch_end(outputs, dataloader_idx, mode='test')

    @rank_zero_only
    def _maybe_save_predictions(self, outputs: List[Dict], mode: str = "val", dataloader_idx: int = 0):
        """
        Save predictions to disk.
        Args:
            outputs: list of outputs
            mode: mode of the model, either 'val' or 'test'
        """

        if not self.cfg.get('save_pred_to_file', None):
            return

        output_file = Path(self.cfg.save_pred_to_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        output_file = output_file.with_suffix(f'.{dataloader_idx}.json')

        manifest = []
        for output in outputs:
            for i in range(len(output[f'{mode}_sample_id'])):
                item = {
                    "sample_id": output[f'{mode}_sample_id'][i],
                    "audio_filepath": output[f'{mode}_audio_filepath'][i],
                    "eou_text": output[f'{mode}_text_gt'][i],
                    "eou_pred_text": output[f'{mode}_text_pred'][i],
                }
                manifest.append(item)
        write_manifest(output_file, manifest)
        logging.info(f"Predictions saved to {output_file}")
        return output_file
