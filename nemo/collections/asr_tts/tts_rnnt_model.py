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

import contextlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch_lightning import Trainer
from transformers import EncodecModel

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import AudioTextBatchWithSpeakerId
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.rnnt_wer import RNNTWER, RNNTDecoding
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.tts.parts.utils.helpers import g2p_backward_compatible_support, process_batch
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import LabelsType, LengthsType, NeuralType
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum


class TextToSpeechTransducerModel(EncDecRNNTModel, ASRBPEMixin):
    class DatasetType(PrettyStrEnum):
        ASR_BPE = "asr_bpe"
        TTS = "tts"

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        self.dataset_type = self.DatasetType(cfg.dataset_type)  # setup before super
        if cfg.encoder.feat_in <= 0:
            cfg.encoder.feat_in = cfg.speaker_embedding_dim + cfg.text_embedding_dim

        # Setup the tokenizer
        self.tokenizer: TokenizerSpec
        self.ds_class = None

        if self.dataset_type == self.DatasetType.ASR_BPE:
            self._setup_tokenizer(cfg.tokenizer)
            vocabulary = self.tokenizer.tokenizer.get_vocab()
            vocabulary_size = len(vocabulary)
        elif self.dataset_type == self.DatasetType.TTS:
            self.ds_class = cfg.train_ds.dataset._target_
            # self.vocab = None
            self._setup_tokenizer_tts(cfg)
            assert self.tokenizer is not None
            vocabulary = self.tokenizer.tokens
            vocabulary_size = len(vocabulary)
        else:
            raise NotImplementedError(f"Unsupported dataset type {self.dataset_type}")

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = 1024  # encodec, TODO: read from config

        super(EncDecRNNTModel, self).__init__(cfg=cfg, trainer=trainer)

        self.text_embeddings = nn.Embedding(vocabulary_size, cfg.text_embedding_dim)
        self.encodec_model: nn.Module = EncodecModel.from_pretrained(
            "facebook/encodec_24khz"
        )  # TODO: store config, load weights
        self.speaker_embeddings = nn.Embedding(self.cfg.n_speakers, self.cfg.speaker_embedding_dim)

        # Initialize components
        self.preprocessor = EncDecRNNTModel.from_config_dict(self.cfg.preprocessor)
        self.encoder = EncDecRNNTModel.from_config_dict(self.cfg.encoder)

        # Update config values required by components dynamically
        with open_dict(self.cfg.decoder):
            self.cfg.decoder.vocab_size = self.cfg.num_osymbols

        with open_dict(self.cfg.joint):
            self.cfg.joint.num_classes = self.cfg.num_osymbols
            self.cfg.joint.vocabulary = ListConfig(
                list(map(lambda token: f"{token} ", range(self.cfg.num_osymbols)))  # mind " " after token
            )
            self.cfg.joint.jointnet.encoder_hidden = self.cfg.model_defaults.enc_hidden
            self.cfg.joint.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden

        self.decoder = EncDecRNNTModel.from_config_dict(self.cfg.decoder)
        self.joint = EncDecRNNTModel.from_config_dict(self.cfg.joint)

        # Setup RNNT Loss
        loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))

        num_classes = self.joint.num_classes_with_blank - 1  # for standard RNNT and multi-blank

        if loss_name == 'tdt':
            num_classes = num_classes - self.joint.num_extra_outputs

        self.loss = RNNTLoss(
            num_classes=num_classes,
            loss_name=loss_name,
            loss_kwargs=loss_kwargs,
            reduction=self.cfg.get("rnnt_reduction", "mean_batch"),
        )

        if hasattr(self.cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecRNNTModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup decoding objects
        self.decoding = RNNTDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, vocabulary=self.joint.vocabulary,
        )
        # Setup WER calculation
        if self._cfg.get('use_cer'):
            logging.warning("use_cer is set to True, but will be ignored. WER = Token Error Rate in this model")
        self.wer = RNNTWER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=False,  # self._cfg.get('use_cer', False)
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Whether to compute loss during evaluation
        if 'compute_eval_loss' in self.cfg:
            self.compute_eval_loss = self.cfg.compute_eval_loss
        else:
            self.compute_eval_loss = True

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Setup optimization normalization (if provided in config)
        self.setup_optim_normalization()

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # Setup encoder adapters (from ASRAdapterModelMixin)
        self.setup_adapters()

    def change_vocabulary(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "transcript": NeuralType(('B', 'T'), LabelsType()),
            "transcript_len": NeuralType(tuple('B'), LengthsType()),
            "speaker_ids": NeuralType(tuple('B'), LengthsType()),  # TODO: speaker type???
        }

    @typecheck()
    def forward(self, transcript: torch.Tensor, transcript_len: torch.Tensor, speaker_ids: torch.Tensor):
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
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        # has_processed_signal = False
        # if not has_processed_signal:
        #     processed_signal, processed_signal_length = self.preprocessor(
        #         input_signal=input_signal, length=input_signal_length,
        #     )
        embed_transcript = self.text_embeddings(transcript).transpose(1, 2)
        batch_size, text_length = transcript.shape[0], transcript.shape[1]
        embed_speakers = self.speaker_embeddings(speaker_ids).unsqueeze(-1).expand(-1, -1, text_length)
        embed_transcript = torch.cat((embed_transcript, embed_speakers), 1)

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            # 'B', 'D', 'T'
            embed_transcript = self.spec_augmentation(input_spec=embed_transcript, length=transcript_len)

        encoded, encoded_len = self.encoder(audio_signal=embed_transcript, length=transcript_len)
        return encoded, encoded_len

    def training_step(self, batch, batch_nb):
        if not isinstance(batch, AudioTextBatchWithSpeakerId) and self.dataset_type != self.DatasetType.TTS:
            raise NotImplementedError("Unsupported batch type")
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if isinstance(batch, AudioTextBatchWithSpeakerId) and self.dataset_type != self.DatasetType.ASR_BPE:
            transcript = batch.transcripts
            transcript_len = batch.transcripts_length
            speaker_ids = batch.speaker_ids
            audio = batch.audio_signal
            audio_len = batch.audio_signal_length
        elif self.dataset_type == self.DatasetType.TTS:
            batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
            transcript = batch_dict.get("text")
            transcript_len = batch_dict.get("text_lens")
            speaker_ids = batch_dict.get("speaker_id", None)
            audio = batch_dict.get("audio")
            audio_len = batch_dict.get("audio_lens")
            semantic_code = batch_dict.get("semantic_code", None)
            semantic_code_len = batch_dict.get("semantic_code_lens", None)
        else:
            raise NotImplementedError("Unsupported batch type")

        # signal, signal_len, transcript, transcript_len = batch

        encoded, encoded_len = self.forward(
            transcript=transcript, transcript_len=transcript_len, speaker_ids=speaker_ids
        )

        if semantic_code is None:
            with torch.no_grad():
                if self.encodec_model.training:
                    self.encodec_model.eval()
                # signal.shape: B, T
                signal_mask = (
                    torch.arange(audio.shape[-1], device=audio.device)[None, :]
                    < audio_len[:, None]
                )
                quantized_signal = self.encodec_model.encode(
                    audio.unsqueeze(1), signal_mask.unsqueeze(1), bandwidth=1.5
                ).audio_codes.squeeze(0)
                quantized_signal = quantized_signal[:, 1].squeeze(1)  # first codebook

                quantized_signal_len = torch.ceil(torch.div(audio_len * 75, 24000)).to(torch.long)
        else:
            quantized_signal = semantic_code
            quantized_signal_len = semantic_code_len
        # logging.warning(f"signal: {signal.shape}, {signal_len}")
        # logging.warning(f"transcript: {transcript.shape}, {transcript_len}")
        # logging.warning(f"transcript encoded: {encoded.shape}, {encoded_len}")
        # logging.warning(f"signal quantized {quantized_signal.shape}, {quantized_signal_len}")

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=quantized_signal, target_length=quantized_signal_len)

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
                log_probs=joint,
                targets=quantized_signal,
                input_lengths=encoded_len,
                target_lengths=quantized_signal_len,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled():
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(encoded, encoded_len, quantized_signal, quantized_signal_len)
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
                transcripts=quantized_signal,
                transcript_lengths=quantized_signal_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled():
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
            self._optim_normalize_txu = [encoded_len.max(), quantized_signal_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError  # TODO
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if not isinstance(batch, AudioTextBatchWithSpeakerId) and self.dataset_type != self.DatasetType.TTS:
            raise NotImplementedError("Unsupported batch type")
        
        if isinstance(batch, AudioTextBatchWithSpeakerId) and self.dataset_type != self.DatasetType.ASR_BPE:
            transcript = batch.transcripts
            transcript_len = batch.transcripts_length
            speaker_ids = batch.speaker_ids
            audio = batch.audio_signal
            audio_len = batch.audio_signal_length
        elif self.dataset_type == self.DatasetType.TTS:
            batch_dict = process_batch(batch, self._train_dl.dataset.sup_data_types_set)
            transcript = batch_dict.get("text")
            transcript_len = batch_dict.get("text_lens")
            speaker_ids = batch_dict.get("speaker_id", None)
            audio = batch_dict.get("audio")
            audio_len = batch_dict.get("audio_lens")
            semantic_code = batch_dict.get("semantic_code", None)
            semantic_code_len = batch_dict.get("semantic_code_lens", None)
        else:
            raise NotImplementedError("Unsupported batch type")
        # signal, signal_len, transcript, transcript_len = batch

        encoded, encoded_len = self.forward(
            transcript=transcript, transcript_len=transcript_len, speaker_ids=speaker_ids
        )
        if semantic_code is None:
            with torch.no_grad():
                if self.encodec_model.training:
                    self.encodec_model.eval()
                # signal.shape: B, T
                signal_mask = (
                    torch.arange(audio_len.shape[-1], device=audio.device)[None, :]
                    < audio_len[:, None]
                )
                quantized_signal = self.encodec_model.encode(
                    audio.unsqueeze(1), signal_mask.unsqueeze(1), bandwidth=1.5
                ).audio_codes.squeeze(0)
                quantized_signal = quantized_signal[:, 1].squeeze(1)  # first codebook

                quantized_signal_len = torch.ceil(torch.div(audio_len * 75, 24000)).to(torch.long)
        else:
            quantized_signal = semantic_code
            quantized_signal_len = semantic_code_len

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:

                decoder, target_length, states = self.decoder(
                    targets=quantized_signal, target_length=quantized_signal_len
                )
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint,
                    targets=quantized_signal,
                    input_lengths=encoded_len,
                    target_lengths=quantized_signal_len,
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(encoded, encoded_len, quantized_signal, quantized_signal_len)
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(
                    targets=quantized_signal, target_length=quantized_signal_len
                )
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=quantized_signal,
                transcript_lengths=quantized_signal_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {
            'test_wer_num': logs['val_wer_num'],
            'test_wer_denom': logs['val_wer_denom'],
            # 'test_wer': logs['val_wer'],
        }
        if 'val_loss' in logs:
            test_logs['test_loss'] = logs['val_loss']
        return test_logs


    def _setup_tokenizer_tts(self, cfg):
        text_tokenizer_kwargs = {}

        if "g2p" in cfg.text_tokenizer:
            # for backward compatibility
            if (
                self._is_model_being_restored()
                and (cfg.text_tokenizer.g2p.get('_target_', None) is not None)
                and cfg.text_tokenizer.g2p["_target_"].startswith("nemo_text_processing.g2p")
            ):
                cfg.text_tokenizer.g2p["_target_"] = g2p_backward_compatible_support(
                    cfg.text_tokenizer.g2p["_target_"]
                )

            g2p_kwargs = {}

            if "phoneme_dict" in cfg.text_tokenizer.g2p:
                g2p_kwargs["phoneme_dict"] = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict', cfg.text_tokenizer.g2p.phoneme_dict,
                )

            if "heteronyms" in cfg.text_tokenizer.g2p:
                g2p_kwargs["heteronyms"] = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms', cfg.text_tokenizer.g2p.heteronyms,
                )

            # for backward compatability
            text_tokenizer_kwargs["g2p"] = instantiate(cfg.text_tokenizer.g2p, **g2p_kwargs)

        # TODO @xueyang: rename the instance of tokenizer because vocab is misleading.
        self.tokenizer = instantiate(cfg.text_tokenizer, **text_tokenizer_kwargs)

    def setup_training_data(self, config):
        if self.dataset_type == self.DatasetType.ASR_BPE:
            super(TextToSpeechTransducerModel, self).setup_training_data(config)
        else:
            self._train_dl = self._setup_dataloader_from_config(config)

    def setup_validation_data(self, config):
        if self.dataset_type == self.DatasetType.ASR_BPE:
            super(TextToSpeechTransducerModel, self).setup_validation_data(config)
        else:
            self._validation_dl = self._setup_dataloader_from_config(config)


    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if self.dataset_type == self.DatasetType.ASR_BPE:
            return self._setup_dataloader_from_config_asr(config=config)
        elif self.dataset_type == self.DatasetType.TTS:
            return self._setup_dataloader_from_config_tts(config=config)
        raise NotImplementedError(f"Support for dataset of {self.dataset_type} type is not implemented")


    def _setup_dataloader_from_config_tts(self, config: Optional[Dict], shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in config or not isinstance(config.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in config or not isinstance(config.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")
        if shuffle_should_be:
            if 'shuffle' not in config.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(config.dataloader_params):
                    config.dataloader_params.shuffle = True
            elif not config.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif config.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        if self.ds_class == "nemo.collections.tts.data.dataset.TTSDataset":
            phon_mode = contextlib.nullcontext()
            if hasattr(self.vocab, "set_phone_prob"):
                phon_mode = self.vocab.set_phone_prob(prob=None if name == "val" else self.vocab.phoneme_probability)

            with phon_mode:
                dataset = instantiate(
                    config.dataset,
                    text_normalizer=self.normalizer,
                    text_normalizer_call_kwargs=self.text_normalizer_call_kwargs,
                    text_tokenizer=self.vocab,
                )
        else:
            dataset = instantiate(config.dataset, text_tokenizer=self.tokenizer,)

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **config.dataloader_params)


    def _setup_dataloader_from_config_asr(self, config: Optional[Dict]):
        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            raise NotImplementedError

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results
