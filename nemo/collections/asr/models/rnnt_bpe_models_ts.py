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

import copy
import json
import os
from selectors import EpollSelector
import tempfile
from copyreg import dispatch_table
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from tqdm.auto import tqdm
import torch
import torch.nn as nn

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.losses.ss_losses.si_snr import SiSNR
from nemo.collections.asr.modules.conv_asr import Conv2dASRDecoderReconstruction
from nemo.collections.asr.metrics.rnnt_wer_bpe import RNNTBPEWER, RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging, model_utils

__all__ = ['TSEncDecRNNTBPEModel']

all_embs = []


class TSEncDecRNNTBPEModel(EncDecRNNTBPEModel):
    """Encoder decoder RNNT-based models with Byte Pair Encoding."""

    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if hasattr(self._cfg, 'speaker_beam') and self._cfg.speaker_beam is not None:
            self.speaker_beam = EncDecRNNTBPEModel.from_config_dict(self._cfg.speaker_beam)

        if hasattr(self._cfg, 'speaker_embeddings') and self._cfg.speaker_embeddings is not None:
            if self._cfg.speaker_embeddings.model_path:
                self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(self._cfg.speaker_embeddings.model_path)
                if self._cfg.speaker_embeddings.freeze_encoder:
                    self.speaker_model.encoder.freeze()
                if self._cfg.speaker_embeddings.freeze_decoder:
                    self.speaker_model.decoder.freeze()
        if hasattr(self._cfg, 'freeze_asr_encoder') and self._cfg.freeze_asr_encoder:
            self.encoder.freeze()
        if hasattr(self._cfg, 'freeze_asr_decoder') and self._cfg.freeze_asr_decoder:
            self.decoder.freeze()

        
        self.decoder_losses = None
        if "loss_list" in self._cfg:
            self.decoder_losses = {}
            self.loss_alphas = {}
            self.start_step = {}
            self.stop_step = {}

            for decoder_loss_name, decoder_loss_cfg in self._cfg.loss_list.items():
                decoder_loss = {
                    'decoder': Conv2dASRDecoderReconstruction(
                        feat_in=self._cfg.d_model,
                        feat_out=self._cfg.preprocessor.features,
                        channels_hidden=self._cfg.d_model,
                    ),
                    'loss' : SiSNR(return_error=False),
                }
                decoder_loss = nn.ModuleDict(decoder_loss)
                self.decoder_losses[decoder_loss_name] = decoder_loss
                self.loss_alphas[decoder_loss_name] = decoder_loss_cfg.get("alpha", 1.0)
                self.start_step[decoder_loss_name] = decoder_loss_cfg.get("start_step", 0)
                self.stop_step[decoder_loss_name] = decoder_loss_cfg.get("stop_step", -1)
            
            self.decoder_losses = nn.ModuleDict(self.decoder_losses)


    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return []

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        d = super().input_types
        d["speaker_embedding"] = NeuralType(('B', 'T'), AudioSignal())
        d['embedding_lengths'] = NeuralType(tuple('B'), LengthsType())
        d["sample_id"] = NeuralType(tuple('B'), LengthsType(), optional=True)
        d["target_signal"] = NeuralType(('B', 'T'), AudioSignal(), optional=True) 
        return d

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        d = super().output_types
        d["target_signal"] = NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation(), optional=True)
        d["target_signal_estimate"] = NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation(), optional=True)
        d["target_signal_length"] = NeuralType(tuple('B'), LengthsType(), optional=True)
        return d

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        speaker_embedding=None,
        embedding_lengths=None,
        target_signal=None,
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
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # process target if present
        if target_signal is not None:
            target_signal, target_signal_length = self.preprocessor(
                input_signal=target_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        if self.speaker_model:
            _, speaker_embedding = self.speaker_model.forward(
                input_signal=speaker_embedding, input_signal_length=embedding_lengths
            )

        mask, mask_len, pre_encoded_audio, pre_encoded_audio_lengths = self.speaker_beam(
            audio_signal=processed_signal,
            length=processed_signal_length,
            emb=speaker_embedding,
        )
        processed_signal = mask * pre_encoded_audio.permute(0, 2, 1)    # [B, D, T]

        # target estimate
        if 'reconstruction' in self.decoder_losses:
            target_signal_estimate = self.decoder_losses['reconstruction']['decoder'](encoder_output=processed_signal)

        encoded, encoded_len, _, _ = self.encoder(audio_signal=processed_signal, length=pre_encoded_audio_lengths)
        
        if target_signal is not None:
            return encoded, encoded_len, target_signal, target_signal_estimate, target_signal_length
        else:
            return encoded, encoded_len, None, None, None

    # PTL-specific methods
    def training_step(self, batch, batch_nb):

        signal, signal_len, transcript, transcript_len, speaker_embedding, embedding_lengths, target_signal = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len, target_signal, target_signal_estimate, target_signal_length = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                speaker_embedding=speaker_embedding,
                embedding_lengths=embedding_lengths,
                target_signal=target_signal if 'reconstruction' in self.decoder_losses else None,
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

            tensorboard_logs = {
                'rnnt_loss': loss_value,
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            tensorboard_logs.update(
                dict([(f"learning_rate_group{i}" , param_group['lr']) for i, param_group in enumerate(self._optimizer.param_groups)])
            )

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(encoded, encoded_len, transcript, transcript_len)
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

            tensorboard_logs = {
                'rnnt_loss': loss_value,
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            tensorboard_logs.update(
                dict([(f"learning_rate_group{i}" , param_group['lr']) for i, param_group in enumerate(self._optimizer.param_groups)])
            )

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        ## other  losses
        #################
        # if 'reconstruction' in self.decoder_losses:
        for dec_loss_name, dec_loss in self.decoder_losses.items():
            # loop through decoders and losses
            if (
                hasattr(self, "trainer")
                and self.trainer is not None
                and (
                    self.trainer.global_step < self.start_step[dec_loss_name] or 
                    (self.trainer.global_step >= self.stop_step[dec_loss_name] and self.stop_step[dec_loss_name] >= 0)
                )
            ):
                continue

            _, _, T = target_signal.shape
            target_signal_estimate = target_signal_estimate[:, :, :T]

            B, D, T = target_signal_estimate.shape
            mask = self.get_mask(target_signal, target_signal_length)
            # target_signal_length = target_signal_length.unsqueeze(-1).expand(-1,D).reshape(-1)
            current_loss_value = dec_loss['loss'](
                target=target_signal.reshape(B, -1).transpose(0, 1).unsqueeze(-1),
                estimate=target_signal_estimate.reshape(B, -1).transpose(0, 1).unsqueeze(-1),
                mask=mask.reshape(B, -1).transpose(0, 1).unsqueeze(-1),
            )
            if dec_loss_name == 'reconstruction':
                current_loss_value = current_loss_value.mean().clamp(-30.0, 999999.0)

            # add to tensorboard
            tensorboard_logs.update({f'{dec_loss_name}_loss': current_loss_value})
            loss_value  = loss_value + self.loss_alphas[dec_loss_name] * current_loss_value
        
        # add total loss
        tensorboard_logs.update({'train_loss': loss_value})
            
        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, speaker_embedding, embedding_lengths, target_signal = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len, target_signal, target_signal_estimate, target_signal_length = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                speaker_embedding=speaker_embedding,
                embedding_lengths=embedding_lengths,
                target_signal=target_signal if 'reconstruction' in self.decoder_losses else None,
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

                tensorboard_logs['val_rnnt_loss'] = loss_value

            self.wer.update(encoded, encoded_len, transcript, transcript_len)
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
                tensorboard_logs['val_rnnt_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log_dict({'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32)})

        ## other  losses
        #################
        for dec_loss_name, dec_loss in self.decoder_losses.items():
            # loop through decoders and losses
            if (
                hasattr(self, "trainer")
                and self.trainer is not None
                and (
                    self.trainer.global_step < self.start_step[dec_loss_name] or 
                    (self.trainer.global_step >= self.stop_step[dec_loss_name] and self.stop_step[dec_loss_name] >= 0)
                )
            ):
                continue

            _, _, T = target_signal.shape
            target_signal_estimate = target_signal_estimate[:, :, :T]

            B, D, T = target_signal_estimate.shape
            mask = self.get_mask(target_signal, target_signal_length)
            # target_signal_length = target_signal_length.unsqueeze(-1).expand(-1,D).reshape(-1)
            current_loss_value = dec_loss['loss'](
                target=target_signal.reshape(B, -1).transpose(0, 1).unsqueeze(-1),
                estimate=target_signal_estimate.reshape(B, -1).transpose(0, 1).unsqueeze(-1),
                mask=mask.reshape(B, -1).transpose(0, 1).unsqueeze(-1),
            )

            if dec_loss_name == 'reconstruction':
                current_loss_value = current_loss_value.mean().clamp(-30.0, 999999.0)

            if loss_value is not None:
                loss_value = loss_value + self.loss_alphas[dec_loss_name] * current_loss_value
            else:
                loss_value = self.loss_alphas[dec_loss_name] * current_loss_value
            
            tensorboard_logs.update(
                {
                    'val_loss' : loss_value,
                    f'val_spectrogram_{dec_loss_name}_loss': current_loss_value,
                }
            )

        return tensorboard_logs


    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_bpe_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            if config['synthetic_generation'] == True:
                if config['dataset'] == 'wsj0':
                    dataset = audio_to_text_dataset.get_dynamic_target_audio_bpe_dataset_wsj(
                        config=config,
                        tokenizer=self.tokenizer,
                        augmentor=augmentor,
                    )
                else:
                    dataset = audio_to_text_dataset.get_dynamic_target_audio_bpe_dataset(
                        config=config,
                        tokenizer=self.tokenizer,
                        augmentor=augmentor,
                    )
            else:
                if config['dataset'] == 'wsj0':
                    dataset = audio_to_text_dataset.get_static_target_audio_bpe_dataset_wsj(
                        config=config,
                        tokenizer=self.tokenizer,
                        augmentor=augmentor,
                    )
                else:
                    dataset = audio_to_text_dataset.get_static_target_audio_bpe_dataset(
                        config=config,
                        tokenizer=self.tokenizer,
                        augmentor=augmentor,
                    )


        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
        return loader

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
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        with open(config['manifest_filepath'], 'r') as fp:
            man_len = len(fp.readlines())
        batch_size = min(config['batch_size'], man_len)
        dl_config = {
            'manifest_filepath': config['manifest_filepath'],
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'augmentor': config.get('augmentor', None),
            'synthetic_generation': config['synthetic_generation'],
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'num_sources': config["num_sources"]
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    @torch.no_grad()
    def transcribe(
        self,
        manifest_filepath: str,
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = None,
        num_sources = None,
    ) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader

        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if manifest_filepath is None:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            self.speaker_beam.freeze()
            self.speaker_model.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there

            if not num_sources:
                num_sources = self._cfg.test_ds.num_sources
            config = {
                'manifest_filepath': manifest_filepath,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'synthetic_generation': False,
                'num_sources': num_sources
            }

            temporary_datalayer = self._setup_transcribe_dataloader(config)
            for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                signal, signal_len, transcript, transcript_len, speaker_embedding, embedding_lengths = test_batch
                logits, logits_len, greedy_predictions = self.forward(
                    input_signal=signal.to(device),
                    input_signal_length=signal_len.to(device),
                    speaker_embedding=speaker_embedding.to(device),
                    embedding_lengths=embedding_lengths.to(device),
                )

                if logprobs:
                    # dump log probs per file
                    for idx in range(logits.shape[0]):
                        lg = logits[idx][: logits_len[idx]]
                        hypotheses.append(lg.cpu().numpy())
                else:
                    current_hypotheses , all_hyp = self.decoding.ctc_decoder_predictions_tensor(
                            logits, decoder_lengths=logits_len, return_hypotheses=return_hypotheses,
                        )

                    if return_hypotheses:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                    hypotheses += current_hypotheses

                del greedy_predictions
                del logits
                del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
                self.speaker_beam.unfreeze()
                self.speaker_model.unfreeze()
            logging.set_verbosity(logging_level)
        return hypotheses


    def get_mask(self, target, target_lengths):
        """
        args:
            target: [B, D, T]
            target_lengths: [B]

        return:
            mask: [B, D, T]
        """
        B, D, T = target.shape
        mask = target.new_ones((B, T))
        for i in range(B):
            mask[i, target_lengths[i]:] = 0

        mask = mask.unsqueeze(1).expand(-1, D, -1)
        return mask

    
    def setup_optimizer_param_groups(self):
        """
            Used to create param groups for the optimizer.
            As an example, this can be used to specify per-layer learning rates:
            optim.SGD([
                        {'params': model.base.parameters()},
                        {'params': model.classifier.parameters(), 'lr': 1e-3}
                        ], lr=1e-2, momentum=0.9)
            See https://pytorch.org/docs/stable/optim.html for more information.
            By default, ModelPT will use self.parameters().
            Override this method to add custom param groups.
            In the config file, add 'optim_param_groups' to support different LRs 
            for different components (unspecified params will use the default LR):
            model:
                optim_param_groups:
                    encoder: 
                        lr: 1e-4
                        momentum: 0.8
                    decoder: 
                        lr: 1e-3
                optim:
                    lr: 3e-3
                    momentum: 0.9   
        """
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        known_groups = []
        param_groups = []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group, group_cfg in param_groups_cfg.items():
                module = getattr(self, group, None)
                if module is None:
                    raise ValueError(f"{group} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(group)
                    new_group = {"params": module.parameters()}
                    for k, v in group_cfg.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

            other_params = []
            for n, p in self.named_parameters():
                is_unknown = True
                for group in known_groups:
                    if n.startswith(group):
                        is_unknown = False
                if is_unknown:
                    other_params.append(p)

            if len(other_params):
                param_groups = [{"params": other_params}] + param_groups
        else:
            param_groups = [{"params": self.parameters()}]

        self._optimizer_param_groups = param_groups
