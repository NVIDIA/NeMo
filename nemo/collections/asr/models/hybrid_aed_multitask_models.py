from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
import torch


import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    PromptedAudioToTextMiniBatch,
)
from nemo.collections.asr.metrics import MultiTaskMetric
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, ASRModuleMixin, ASRTranscriptionMixin, InterCTCMixin
from nemo.collections.asr.parts.mixins.transcription import (
    GenericTranscriptionType,
    InternalTranscribeConfig,
    TranscribeConfig,
)
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import process_aed_timestamp_outputs
from nemo.collections.common import tokenizers
from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import (
    AudioSignal,
    ChannelType,
    LabelsType,
    LengthsType,
    LogprobsType,
    MaskType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging, model_utils

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector


class EncDecHybridMultiTaskCTCModel(EncDecMultiTaskModel, InterCTCMixin):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        super().__init__(cfg=cfg, trainer=trainer)

        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # lang_ids = sorted(self.tokenizer.tokens_to_ids([f"<|{lang}|>" for lang in self.cfg.langs]))
        # self.ids_to_idx = {id_: idx for idx, id_ in enumerate(lang_ids)}

        raw_ids = torch.tensor(
            self.tokenizer.tokens_to_ids([f"<|{lang}|>" for lang in self.cfg.langs]),
            dtype=torch.long
        )
        raw_ids, _ = raw_ids.sort()

        max_id = int(raw_ids[-1].item())

        # print('Raw ids: ', raw_ids)
        # print('Max id: ', max_id)
        # print('--------------------------------')
        lut = torch.full((max_id+1,), -1, dtype=torch.long)
        lut[raw_ids] = torch.arange(raw_ids.size(0), dtype=torch.long)

        self.register_buffer("id2idx_lut", lut)

        # print('Id2idx lut: ', self.id2idx_lut)

        with open_dict(self.cfg.aux_ctc):
            if self.tokenizer_type == "agg":
                self.cfg.aux_ctc.decoder.vocabulary = ListConfig(vocabulary)
            else:
                self.cfg.aux_ctc.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

            if "feat_in" not in self.cfg.aux_ctc.decoder or (
                not self.cfg.aux_ctc.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self.cfg.aux_ctc.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self.cfg.aux_ctc.decoder or not self.cfg.aux_ctc.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.aux_ctc.decoder.num_classes < 1 and self.cfg.aux_ctc.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.aux_ctc.decoder.num_classes, len(self.cfg.aux_ctc.decoder.vocabulary)
                    )
                )
                self.cfg.aux_ctc.decoder["num_classes"] = len(self.cfg.aux_ctc.decoder.vocabulary)

        self.ctc_decoder = EncDecHybridMultiTaskCTCModel.from_config_dict(self.cfg.aux_ctc.decoder)
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)

        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
        )

        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

        self.ctc_decoding = CTCDecoding(self.cfg.aux_ctc.decoding, vocabulary=self.ctc_decoder.vocabulary)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "aed"

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='ctc_decoder', loss_name='ctc_loss', wer_name='ctc_wer')

        lang_embedding_dim = self.cfg.get('lang_embedding_dim', int(self.encoder._feat_out / 2))
        self.src_lang_embeddings = torch.nn.Embedding(num_embeddings=self.cfg.get('num_src_langs', 1), embedding_dim=lang_embedding_dim)
        self.tgt_lang_embeddings = torch.nn.Embedding(num_embeddings=self.cfg.get('num_tgt_langs', 1), embedding_dim=lang_embedding_dim)


        if self.cfg.get('use_prompt_projection', False) or self.cfg.aux_ctc.get('use_gated_fusion', False):
            self.prompt_fusion = torch.nn.Linear(2 * self.encoder._feat_out, self.encoder._feat_out)
        else:
            self.prompt_fusion = torch.nn.Linear(2 * lang_embedding_dim + self.encoder._feat_out, self.encoder._feat_out)

        if self.cfg.get('use_prompt_projection', False) or self.cfg.aux_ctc.get('use_gated_fusion', False):
            self.prompt_projection = torch.nn.Linear(2 * lang_embedding_dim, self.encoder._feat_out)

    def change_decoding_strategy(self, decoding_cfg: DictConfig, decoder_type: str = None, verbose: bool = True):
        """
        Changes decoding strategy used during Multi Task decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
        """

        if decoder_type is None or decoder_type == 'aed':
            self.cur_decoder = "aed"
            return super().change_decoding_strategy(decoding_cfg=decoding_cfg, verbose=verbose)

        assert decoder_type == 'ctc' and hasattr(self, 'ctc_decoder')
        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.ctc_decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.ctc_wer.use_cer,
            log_prediction=self.ctc_wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.decoding = decoding_cfg

        self.cur_decoder = "ctc"
        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}")

        return None

    def ctc_forward(self, enc_states: torch.Tensor, prompt: torch.Tensor):

        # print('Batch prompt: ', batch.prompt)

        # print('Source Lang IDs: ', batch.prompt[:, 4].long())
        # print('Target Lang IDs: ', batch.prompt[:, 5].long())

        src_lang_ids_dec = self.tokenizer.ids_to_tokens(prompt[:, 4].long())
        tgt_lang_ids_dec = self.tokenizer.ids_to_tokens(prompt[:, 5].long())

        batch_src_lang_ids = self.id2idx_lut[prompt[:, 4].long()]
        batch_tgt_lang_ids = self.id2idx_lut[prompt[:, 5].long()]
        # print('Src lang ids dec: ', src_lang_ids_dec)
        # print('Tgt lang ids dec: ', tgt_lang_ids_dec)

        # print('Batch src lang ids: ', batch_src_lang_ids)
        # print('Batch tgt lang ids: ', batch_tgt_lang_ids)

        # batch_src_lang_ids = [self.ids_to_idx[id_] for id_ in batch.prompt[:, 4].tolist()]
        # batch_tgt_lang_ids = [self.ids_to_idx[id_] for id_ in batch.prompt[:, 5].tolist()]

        # print('Src Embeddings: ', self.src_lang_embeddings)
        # print('Tgt Embeddings: ', self.tgt_lang_embeddings)


        # print('SRC ID: ', batch_src_lang_ids, "SRC ID TOKEN: ", self.tokenizer.ids_to_tokens(batch_src_lang_ids))
        # print('TGT ID: ', batch_tgt_lang_ids, "TGT ID TOKEN: ", self.tokenizer.ids_to_tokens(batch_tgt_lang_ids))

        # print('Batch src lang ids: ', batch_src_lang_ids)
        # print('Batch tgt lang ids: ', batch_tgt_lang_ids)

        src_lang_embeddings = self.src_lang_embeddings(batch_src_lang_ids)
        tgt_lang_embeddings = self.tgt_lang_embeddings(batch_tgt_lang_ids)

        # print('Src lang embeddings shape: ', src_lang_embeddings.shape)
        # print('Tgt lang embeddings shape: ', tgt_lang_embeddings.shape)

        prompt_hidden = torch.cat([src_lang_embeddings, tgt_lang_embeddings], dim=1)

        # print('Prompt hidden shape: ', prompt_hidden.shape)
        # prompt_hidden = self.prompt_projection(prompt_hidden)

        # print('Prompt hidden after projection shape: ', prompt_hidden.shape)

        # print('Enc states shape: ', enc_states.shape)

        if self.cfg.get('use_prompt_projection', False) or self.cfg.aux_ctc.get('use_gated_fusion', False):
            prompt_hidden = self.prompt_projection(prompt_hidden)

        p_expanded = prompt_hidden.unsqueeze(1)
        prompt_broadcast = p_expanded.expand(-1, enc_states.size(1), -1)

        # print('Prompt broadcast shape: ', prompt_broadcast.shape)

        enc_states_lang = torch.cat([enc_states, prompt_broadcast], dim=-1)

        # print('Combined Enc states shape: ', enc_states_lang.shape)

        enc_states_lang = self.prompt_fusion(enc_states_lang)

        if self.cfg.aux_ctc.get('use_gated_fusion', False):
            # print('Using gated fusion')
            gate = torch.sigmoid(enc_states_lang)
            enc_states_lang = gate * enc_states + (1 - gate) * prompt_broadcast

        enc_states_lang = enc_states_lang.permute(0, 2, 1)

        # print('Combined Enc states shape after fusion: ', enc_states_lang.shape)

        # print("CTC decoder: ", self.ctc_decoder)
        
        log_probs = self.ctc_decoder(encoder_output=enc_states_lang)

        # print('Log probs shape: ', log_probs.shape)
        

        return log_probs
        

    # PTL-specific methods
    def training_step(self, batch: PromptedAudioToTextMiniBatch, batch_nb):
        if batch is None:
            return torch.tensor([0.0])

        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1

        # print('Transcript: ', batch.transcript)
        # print('Transcript length: ', batch.transcript_lens)

        # print('Prompted transcript: ', batch.prompted_transcript)
        # print('Prompted transcript length: ', batch.prompted_transcript_lens)

        num_frames = batch.audio_lens.sum().float()
        num_tokens = batch.prompted_transcript_lens.sum().float()
        tot_frames = torch.as_tensor(batch.audio.numel(), device=num_frames.device, dtype=torch.float)
        tot_tokens = torch.as_tensor(batch.prompted_transcript.numel(), device=num_frames.device, dtype=torch.float)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=input_ids_lens,
        )
        # Mask components: 1) discard padding  &  2) discard prompt (notice the negation)
        # For a full decoder sequence O with len M, the loss mask skips the first element,
        # covering the remaining M-1 elements - hence we subtract 1 from prompt lens to account BOS.
        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
        else:
            loss_mask = None
        loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)

        ctc_metrics = {}
        if self.ctc_loss_weight > 0:
            ctc_metrics['transf_loss'] = loss

            ctc_log_probs = self.ctc_forward(enc_states=enc_states, prompt=batch.prompt)
            ctc_loss = self.ctc_loss(
                log_probs=ctc_log_probs, targets=batch.transcript, input_lengths=encoded_len, target_lengths=batch.transcript_lens
            )
            loss = (1 - self.ctc_loss_weight) * loss + self.ctc_loss_weight * ctc_loss
            ctc_metrics['train_ctc_loss'] = ctc_loss
            ctc_metrics['train_loss'] = loss

            if self.cfg.get('compute_wer', False):
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                ctc_metrics['training_batch_wer_ctc'] = ctc_wer

        # Train step evaluation. From other asr models.
        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1
        metric_dict = (
            self.metric.eval(
                batch=batch,
                predictions=enc_states,
                predictions_lengths=encoded_len,
                predictions_mask=enc_mask,
                prefix="training_batch",
            )
            if (batch_nb + 1) % log_every_n_steps == 0
            else {}
        )

        metric_dict.update(
            {
                'train_loss': loss,
                'learning_rate': torch.as_tensor(self._optimizer.param_groups[0]['lr']),
                'batch_size': torch.as_tensor(batch.audio.shape[0]),
                'num_frames': num_frames,
                'num_tokens': num_tokens,
                'input_to_padding_ratio': num_frames / tot_frames,
                'output_to_padding_ratio': num_tokens / tot_tokens,
            }
        )

        metric_dict.update(ctc_metrics)
        # if (batch_nb + 1) % log_every_n_steps == 0:
        #     print('Metric dict: ', metric_dict)
        # print('CTC Loss: ', ctc_loss)
        return {"loss": loss, "log": metric_dict}

    def validation_pass(self, batch: PromptedAudioToTextMiniBatch, batch_idx, dataloader_idx=0, eval_mode="val"):
        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=batch.prompted_transcript_lens,
        )

        # Mask components: 1) discard padding  &  2) discard prompt (notice the negation)
        # For a full decoder sequence O with len M, the loss mask skips the first element,
        # covering the remaining M-1 elements - hence we subtract 1 from prompt lens to account BOS.
        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
            num_measurements = loss_mask.long().sum()
        else:
            loss_mask = None
            num_measurements = transf_log_probs.shape[0] * transf_log_probs.shape[1]

        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)
        self.val_loss(loss=transf_loss, num_measurements=num_measurements)

        metric_dict = self.metric.eval(
            batch=batch,
            predictions=enc_states,
            predictions_lengths=encoded_len,
            predictions_mask=enc_mask,
            prefix=eval_mode,
            return_all_metrics=True,  # Need all metrics for computation at end of cycle.
        )
        metric_dict[f"{eval_mode}_loss"] = transf_loss
        return metric_dict


    def _transcribe_output_processing(self, outputs, trcfg: MultiTaskTranscriptionConfig) -> GenericTranscriptionType:
        """
        Internal function to process the model's outputs to return the results to the user. This function is called by
        `transcribe()` and `transcribe_generator()` to process the model's outputs.

        Args:
            outputs: The model's outputs that are processed by `_transcribe_forward()`.
            trcfg: The transcription config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The output can be a list of
            objects, list of list of objects.
            Its type is defined in `TranscriptionReturnType`.
        """
        if self.cur_decoder == "aed":
            return super()._transcribe_output_processing(outputs, trcfg)
        else:
            log_probs = outputs.pop('log_probs')
            encoded_len = outputs.pop('encoded_lengths')
            enc_states = outputs.pop('encoder_states')
            enc_mask = outputs.pop('encoder_mask')
            decoder_input_ids = outputs.pop('decoder_input_ids')

            del log_probs, enc_mask, decoder_input_ids

            ctc_log_probs = self.ctc_forward(enc_states=enc_states, prompt=decoder_input_ids)

            hypotheses = self.ctc_decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=ctc_log_probs,
                    decoder_lengths=encoded_len,
                    return_hypotheses=trcfg.return_hypotheses,
                    )

            del enc_states, encoded_len, ctc_log_probs

            hypotheses = process_aed_timestamp_outputs(
                hypotheses, self.encoder.subsampling_factor, self.cfg['preprocessor']['window_stride']
            )

            return hypotheses
        
    @classmethod
    def restore_from(cls, 
                    restore_path: str, 
                    override_config_path: Optional[Union[OmegaConf, str]] = None, 
                    map_location: Optional[torch.device] = None, 
                    strict: bool = True, 
                    return_config: bool = False, 
                    save_restore_connector: SaveRestoreConnector = None, 
                    trainer: Optional[Trainer] = None, 
                    validate_access_integrity: bool = True):

        try:
            return super().restore_from(restore_path, override_config_path, map_location, strict, return_config, save_restore_connector, trainer, validate_access_integrity)
        except Exception as e:
            print(e)
            print('--------------------------------')
            print('Trying to restore from EncDecMultiTaskModel')
            print('--------------------------------')
            return EncDecMultiTaskModel.restore_from(restore_path, override_config_path, map_location, strict, return_config, save_restore_connector, trainer, validate_access_integrity)