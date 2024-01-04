# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torchmetrics.text.bleu import _bleu_score_compute  # TODO: Move to asr_model pass
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.metrics.bleu import BLEU
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.modules.transformer import TransformerEncoder
from nemo.collections.asr.modules.transformer.transformer_decoding import TransformerBPEConfig, TransformerDecoding
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.common.parts import transformer_weights_init
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
from nemo.utils import logging

try:
    from nemo.collections.nlp.modules.common import TokenClassifier
    from nemo.collections.nlp.modules.common.lm_utils import get_transformer

    NLP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLP_AVAILABLE = False
    logging.warning("Could not import NeMo NLP collection which is required for speech translation model.")

__all__ = ['EncDecTransfModelBPE']


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


class EncDecTransfModelBPE(ASRModel, ExportableEncDecModel, ASRBPEMixin):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup audio preprocessor
        self.preprocessor = EncDecTransfModelBPE.from_config_dict(self.cfg.preprocessor)

        # Setup specaug
        if hasattr(self.cfg, 'spec_augment') and self.cfg.spec_augment is not None:
            self.spec_augmentation = EncDecTransfModelBPE.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup audio encoder
        self.encoder = EncDecTransfModelBPE.from_config_dict(self.cfg.encoder)

        # # Setup decoder module
        # self.decoder = EncDecTransfModelBPE.from_config_dict(self._cfg.decoder)

        # Add projection layer if encoder and decoder differ in hidden size
        if self.cfg.encoder['d_model'] != self.cfg.transf_decoder['hidden_size']:
            self.adapter = torch.nn.Linear(self.cfg.encoder['d_model'], self.cfg.transf_decoder['hidden_size'])
        else:
            self.adapter = torch.nn.Identity()

        # Whether to add Transformer Encoder block between Conformer and Transformer Decoder
        self.use_transf_encoder = False
        transf_encoder_cfg_dict = OmegaConf.to_container(cfg.get('transf_encoder'))
        if transf_encoder_cfg_dict['num_layers'] > 0:
            self.use_transf_encoder = True

            self.transf_encoder = TransformerEncoder(
                num_layers=transf_encoder_cfg_dict['num_layers'],
                hidden_size=transf_encoder_cfg_dict['hidden_size'],
                inner_size=transf_encoder_cfg_dict['inner_size'],
                mask_future=False,
                num_attention_heads=transf_encoder_cfg_dict['num_attention_heads'],
                attn_score_dropout=transf_encoder_cfg_dict['attn_score_dropout'],
                attn_layer_dropout=transf_encoder_cfg_dict['attn_layer_dropout'],
                ffn_dropout=transf_encoder_cfg_dict['ffn_dropout'],
                pre_ln=transf_encoder_cfg_dict.get('pre_ln', True),
                pre_ln_final_layer_norm=transf_encoder_cfg_dict.get('pre_ln_final_layer_norm', True),
            )
            std_init_range = 1 / transf_encoder_cfg_dict['hidden_size'] ** 0.5
            self.transf_encoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        # Transformer decoder
        transf_decoder_cfg_dict = OmegaConf.to_container(cfg.get('transf_decoder'))
        vocab_size = 8 * ceil(self.tokenizer.vocab_size / 8)
        transf_decoder_cfg_dict['vocab_size'] = vocab_size
        library = transf_decoder_cfg_dict.pop('library', 'nemo')
        model_name = transf_decoder_cfg_dict.pop('model_name', None)
        pretrained = transf_decoder_cfg_dict.pop('pretrained', False)
        self.transf_decoder = get_transformer(
            library=library,
            model_name=model_name,
            pretrained=pretrained,
            config_dict=transf_decoder_cfg_dict,
            encoder=False,
            pre_ln_final_layer_norm=transf_decoder_cfg_dict.get("pre_ln_final_layer_norm", False),
        )

        # Setup classifier
        self.log_softmax = TokenClassifier(
            hidden_size=self.transf_decoder.hidden_size,
            num_classes=vocab_size,
            activation=self.cfg.head.activation,
            log_softmax=self.cfg.head.log_softmax,
            dropout=self.cfg.head.dropout,
            use_transformer_init=self.cfg.head.use_transformer_init,
        )
        self.log_softmax.mlp.layer0.weight = self.transf_decoder.embedding.token_embedding.weight
        std_init_range = 1 / self.transf_decoder.hidden_size ** 0.5
        self.transf_decoder.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        # Define autoregressive CE loss
        self.loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id, label_smoothing=self.cfg.label_smoothing)

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)
        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(TransformerBPEConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg
        self.decoding = TransformerDecoding(
            decoding_cfg=self.cfg.decoding,
            transformer_decoder=self.transf_decoder,
            classifier=self.log_softmax,
            tokenizer=self.tokenizer,
        )

        # Setup metric. Need unique metric per dataloader
        self.bleu = BLEU(
            decoding=self.decoding,
            log_prediction=self.cfg.get("log_prediction", False),
            dist_sync_on_step=True,
            tokenize=self.cfg.get("bleu_tokenize", "13a"),
        )

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
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
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses or logprobs:
            raise NotImplemented("Return hypotheses is not available with this model.")

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

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
            self.transf_decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=not verbose):
                    log_probs, encoded_len, enc_states, enc_mask = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )

                    beam_hypotheses = (
                        self.beam_search(
                            encoder_hidden_states=enc_states, encoder_input_mask=enc_mask, return_beam_scores=False
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    beam_hypotheses = [self.tokenizer.ids_to_text(hyp) for hyp in beam_hypotheses]

                    # TODO: add support for return_hypotheses=True @AlexGrinch
                    # if return_hypotheses:
                    #     # dump log probs per file
                    #     for idx in range(logits.shape[0]):
                    #         current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                    hypotheses += beam_hypotheses

                    del test_batch, log_probs, encoded_len, enc_states, enc_mask
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.transf_decoder.unfreeze()
            logging.set_verbosity(logging_level)

        return hypotheses

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

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

        shuffle = config['shuffle']
        if config.get('is_tarred', False):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):

        # create audio-only data loader
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the
        # dataloader is the total number of samples rather than the number of batches,
        # and this messes up the tqdm progress bar. So we set the number of steps manually
        # (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane,
            # i.e. <= # training batches, and don't change it. Otherwise, adjust
            # batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.
        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.
        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.
        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "transcript": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "transcript_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "transf_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "encoder_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        transcript=None,
        transcript_length=None,
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
                input_signal=input_signal, length=input_signal_length
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.adapter(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
            )
            log_probs = self.log_softmax(hidden_states=dec_states)
        return log_probs, encoded_len, enc_states, enc_mask

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, enc_states, enc_mask = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        else:
            log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )

        loss_val = self.loss(log_probs=log_probs, labels=labels)
        tensorboard_logs = {
            'train_loss': loss_val,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_nb + 1) % log_every_n_steps == 0:
            self.bleu.update(
                encoder_output=enc_states,
                encoder_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            tensorboard_logs.update({'training_batch_bleu': self.bleu.compute()["bleu"]})
            self.bleu.reset()

        return {'loss': loss_val, 'log': tensorboard_logs}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, enc_states, enc_mask = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        else:
            log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        loss_val = self.loss(log_probs=log_probs, labels=labels)

        output_dict = {f'{eval_mode}_loss': loss_val}
        self.bleu.update(
            encoder_output=enc_states, encoder_lengths=encoded_len, targets=transcript, targets_lengths=transcript_len,
        )
        output_dict.update(self.bleu.compute(prefix=eval_mode))
        self.bleu.reset()
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))
        return output_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    # TODO: move logic to ASR_Model.
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_bleu_pred = torch.stack([x['val_bleu_pred'] for x in outputs]).sum()
        val_bleu_targ = torch.stack([x['val_bleu_target'] for x in outputs]).sum()
        val_bleu_num = torch.stack([x['val_bleu_num'] for x in outputs]).sum(dim=0)
        val_bleu_denom = torch.stack([x['val_bleu_denom'] for x in outputs]).sum(dim=0)
        tensorboard_logs = {
            'val_loss': val_loss_mean,
            'val_bleu': _bleu_score_compute(
                val_bleu_pred,
                val_bleu_targ,
                val_bleu_num,
                val_bleu_denom,
                self.bleu.n_gram,
                self.bleu.weights,
                self.bleu.smooth,
            ),
        }
        self.bleu.reset()

        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, eval_mode="test")

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': test_loss_mean}
        bleu_score = {f"test_bleu": self.bleu.compute()}
        tensorboard_logs.update(bleu_score)
        self.bleu.reset()

        return {'test_loss': test_loss_mean, 'log': tensorboard_logs}

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

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
        batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer
