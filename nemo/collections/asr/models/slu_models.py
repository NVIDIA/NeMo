# ! /usr/bin/python
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

from pathlib import Path
from typing import Dict, Optional
from urllib.error import HTTPError

import torch
from omegaconf import DictConfig, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.slu_losses import SmoothedNLLLoss
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.slu_utils import SearcherConfig, SequenceGenerator, get_seq_mask
from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.core import adapter_mixins
from nemo.core.classes.common import Serialization, typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging


class SLUIntentSlotBPEModel(EncDecCTCModelBPE):
    def __init__(self, cfg: DictConfig, trainer=None):
        if hasattr(cfg, "adapter") and getattr(cfg.adapter, "enabled", False):
            logging.info("Using adapters...")
            with open_dict(cfg):
                adapter_metadata = adapter_mixins.get_registered_adapter(cfg.encoder._target_)
                if adapter_metadata is not None:
                    cfg.encoder._target_ = adapter_metadata.adapter_class_path

        super().__init__(cfg=cfg, trainer=trainer)

        # Init encoder from pretrained model
        if self.cfg.pretrained_encoder.model is not None:
            if Path(self.cfg.pretrained_encoder.model).is_file():
                logging.info(f"Loading pretrained encoder from local: {self.cfg.pretrained_encoder.model}")
                pretraind_model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.restore_from(
                    restore_path=self.cfg.pretrained_encoder.model, map_location=torch.device("cpu")
                )
                self.encoder.load_state_dict(pretraind_model.encoder.state_dict(), strict=False)
                del pretraind_model
            else:
                logging.info(f"Loading pretrained encoder from NGC: {self.cfg.pretrained_encoder.model}")
                try:
                    pretraind_model = nemo_asr.models.SpeechEncDecSelfSupervisedModel.from_pretrained(
                        model_name=self.cfg.pretrained_encoder.model, map_location=torch.device("cpu")
                    )
                    self.encoder.load_state_dict(pretraind_model.encoder.state_dict(), strict=False)
                    del pretraind_model
                except HTTPError:
                    logging.warning(f"Unable to load pretrained model: {self.cfg.pretrained_encoder.model}, skipped.")
        else:
            logging.info("Not using pretrained encoder.")

        if self.cfg.pretrained_encoder.freeze:
            logging.info("Freezing encoder...")
            self.encoder.freeze()

        if hasattr(cfg, "adapter") and getattr(cfg.adapter, "enabled", False):
            logging.info("Setting up adapters...")
            adapter_cfg = LinearAdapterConfig(
                in_features=self.cfg.encoder.d_model,  # conformer specific model dim. Every layer emits this dim at its output.
                dim=cfg.adapter.adapter_dim,  # the bottleneck dimension of the adapter
                activation=cfg.adapter.adapter_activation,  # activation used in bottleneck block
                norm_position=cfg.adapter.adapter_norm_position,  # whether to use LayerNorm at the beginning or the end of the adapter
            )
            try:
                self.add_adapter(name=cfg.adapter.adapter_name, cfg=adapter_cfg)
            except ValueError:
                logging.warning(f"Adapter name {cfg.adapter.adapter_name} already exists, skipping.")
            self.set_enabled_adapters(name=cfg.adapter.adapter_name, enabled=True)
            self.encoder.freeze()
            self.unfreeze_enabled_adapters()

        self.vocabulary = self.tokenizer.tokenizer.get_vocab()
        vocab_size = len(self.vocabulary)

        # Create embedding layer
        self.cfg.embedding["vocab_size"] = vocab_size
        self.embedding = Serialization.from_config_dict(self.cfg.embedding)

        # Create token classifier
        self.cfg.classifier["num_classes"] = vocab_size
        self.classifier = Serialization.from_config_dict(self.cfg.classifier)

        self.loss = SmoothedNLLLoss(label_smoothing=self.cfg.get("loss.label_smoothing", 0.0))

        self.searcher = SequenceGenerator(
            cfg=cfg.searcher,
            embedding=self.embedding,
            decoder=self.decoder,
            log_softmax=self.classifier,
            tokenizer=self.tokenizer,
        )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "target_semantics": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "target_semantics_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType(), optional=True),
            "lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType(), optional=True),
        }

    def set_decoding_strategy(self, cfg: SearcherConfig):
        max_len = getattr(self.searcher, "generator.max_seq_length", cfg.max_sequence_length)
        max_delta = getattr(self.searcher, "generator.max_delta_length", cfg.max_delta_length)
        cfg.max_sequence_length = max_len
        cfg.max_delta_length = max_delta
        self.searcher = SequenceGenerator(cfg, self.embedding, self.decoder, self.classifier, self.tokenizer)

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        target_semantics=None,
        target_semantics_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        """
        Forward pass of the model.

        Params:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            target_semantics: Tensor that represents a batch of semantic tokens, of shape [B, L].
            target_semantics_length: Vector of length B, that contains the individual lengths of the semantic
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the output sequence after decoder, of shape [B].
            3) The token predictions of the model of shape [B, T].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoded.transpose(1, 2)  # BxDxT -> BxTxD
        encoded_mask = get_seq_mask(encoded, encoded_len)

        if target_semantics is None:  # in inference-only mode
            predictions = self.searcher(encoded, encoded_mask)
            return None, None, predictions

        bos_semantics_tokens = target_semantics[:, :-1]
        bos_semantics = self.embedding(bos_semantics_tokens)
        bos_semantics_mask = get_seq_mask(bos_semantics, target_semantics_length - 1)

        decoded = self.decoder(
            encoder_states=encoded,
            encoder_mask=encoded_mask,
            decoder_states=bos_semantics,
            decoder_mask=bos_semantics_mask,
        )
        log_probs = self.classifier(decoded)

        predictions = log_probs.argmax(dim=-1, keepdim=False)

        pred_len = self.searcher.get_seq_length(predictions)
        return log_probs, pred_len, predictions

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        if len(batch) == 4:
            signal, signal_len, semantics, semantics_len = batch
        else:
            signal, signal_len, semantics, semantics_len, sample_id = batch

        log_probs, pred_len, predictions = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            target_semantics=semantics,
            target_semantics_length=semantics_len,
        )

        eos_semantics = semantics[:, 1:]
        eos_semantics_len = semantics_len - 1  # subtract 1 for eos tokens

        loss_value = self.loss(log_probs=log_probs, targets=eos_semantics, lengths=eos_semantics_len)

        tensorboard_logs = {'train_loss': loss_value.item()}
        if len(self._optimizer.param_groups) == 1:
            tensorboard_logs['learning_rate'] = self._optimizer.param_groups[0]['lr']
        else:
            for i, group in enumerate(self._optimizer.param_groups):
                tensorboard_logs[f'learning_rate_g{i}'] = group['lr']

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=predictions,
                targets=eos_semantics,
                predictions_lengths=pred_len,
                target_lengths=eos_semantics_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def predict(
        self, input_signal, input_signal_length, processed_signal=None, processed_signal_length=None, dataloader_idx=0
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoded.transpose(1, 2)  # BxDxT -> BxTxD
        encoded_mask = get_seq_mask(encoded, encoded_len)

        pred_tokens = self.searcher(encoded, encoded_mask)
        predictions = self.searcher.decode_semantics_from_tokens(pred_tokens)
        return predictions

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 4:
            signal, signal_len, semantics, semantics_len = batch
        else:
            signal, signal_len, semantics, semantics_len, sample_id = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, pred_len, predictions = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                target_semantics=semantics,
                target_semantics_length=semantics_len,
            )
        else:
            log_probs, pred_len, predictions = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                target_semantics=semantics,
                target_semantics_length=semantics_len,
            )

        eos_semantics = semantics[:, 1:]
        eos_semantics_len = semantics_len - 1  # subtract 1 for bos&eos tokens

        loss_value = self.loss(log_probs=log_probs, targets=eos_semantics, lengths=eos_semantics_len)

        self._wer.update(
            predictions=predictions,
            targets=eos_semantics,
            predictions_lengths=pred_len,
            target_lengths=eos_semantics_len,
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()

        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }
