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

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
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

__all__ = ['TSEncDecCTCModelBPE']

all_embs = []


class TSEncDecCTCModelBPE(EncDecCTCModelBPE):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if hasattr(self._cfg, 'speaker_beam') and self._cfg.speaker_beam is not None:
            self.speaker_beam = EncDecCTCModelBPE.from_config_dict(self._cfg.speaker_beam)
        # self.fuse = torch.nn.Linear(self._cfg.speaker_embeddings.feature_dim, self._cfg.speaker_beam.feat_in)

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

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return EncDecCTCModelBPE.list_available_models()

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        d = super().input_types
        d["speaker_embedding"] = NeuralType(('B', 'T'), AudioSignal())
        d['embedding_lengths'] = NeuralType(tuple('B'), LengthsType())
        d["sample_id"] = d.pop("sample_id")
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
    ):
        """
        Forward pass of the model.

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
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
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

        if self.speaker_model:
            _, speaker_embedding = self.speaker_model.forward(
                input_signal=speaker_embedding, input_signal_length=embedding_lengths
            )

        # fuse processed_signal <- processed_signal + speaker_embedding
        # emb_proj = self.fuse(speaker_embedding).unsqueeze(-1)
        # processed_signal = processed_signal + emb_proj
        mask, mask_len, pre_encoded_audio, pre_encoded_audio_lengths = self.speaker_beam(audio_signal=processed_signal, length=processed_signal_length, emb=speaker_embedding)
        processed_signal = mask * pre_encoded_audio.permute(0, 2, 1)
        encoded, encoded_len, _, _ = self.encoder(audio_signal=processed_signal, length=pre_encoded_audio_lengths)
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, greedy_predictions

    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len, speaker_embedding, embedding_lengths = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len, speaker_embedding=speaker_embedding
            )
        else:
            log_probs, encoded_len, predictions = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                speaker_embedding=speaker_embedding,
                embedding_lengths=embedding_lengths,
            )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_nb + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=predictions,
                targets=transcript,
                target_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, speaker_embedding, embedding_lengths = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                speaker_embedding=speaker_embedding,
                embedding_lengths=embedding_lengths,
            )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        self._wer.update(
            predictions=predictions, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()
        return {
            'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer,
        }

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
                dataset = audio_to_text_dataset.get_dynamic_target_audio_bpe_dataset(
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
