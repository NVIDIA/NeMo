# +
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
import itertools
import json
import random
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import ChainDataset, DataLoader
from tqdm.auto import tqdm
from sacrebleu import corpus_bleu

import editdistance

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.data.audio_to_text import _speech_collate_fn
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer_bpe import WERBPE, CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.features import normalize_batch, clean_spectrogram_batch
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.nlp.models.machine_translation import MTEncDecModel
from nemo.collections.nlp.data.data_utils.data_preprocessing import bitext_collate_fn
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel

from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import (
    AudioSignal,
    LabelsType,
    LengthsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
    MaskType,
    ChannelType,
)
from nemo.utils import logging

from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.losses import NLLLoss, SmoothedCrossEntropyLoss
from nemo.collections.common.parts import transformer_weights_init
from nemo.collections.common.metrics import GlobalAverageLossMetric
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_transformer
from nemo.collections.nlp.modules.common.transformer import (
    TransformerEncoder,
    BeamSearchSequenceGenerator,
)

__all__ = ['EncDecTransfModelBPE']


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


class EncDecTransfModelBPE(ASRModel, ExportableEncDecModel, ASRBPEMixin):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        self.encoder_tokenizer_library = cfg.encoder_tokenizer.get('library', 'yttm')
        self.decoder_tokenizer_library = cfg.decoder_tokenizer.get('library', 'yttm')

        encoder_tokenizer_model, decoder_tokenizer_model, encoder_vocab_file = None, None, None
        if cfg.encoder_tokenizer.get('tokenizer_model') is not None:
            encoder_tokenizer_model = self.register_artifact(
                "encoder_tokenizer.tokenizer_model", cfg.encoder_tokenizer.get('tokenizer_model')
            )

        if cfg.decoder_tokenizer.get('tokenizer_model') is not None:
            decoder_tokenizer_model = self.register_artifact(
                "decoder_tokenizer.tokenizer_model", cfg.decoder_tokenizer.get('tokenizer_model')
            )

        if cfg.encoder_tokenizer.get('vocab_file') is not None:
            encoder_vocab_file = (
                self.register_artifact("encoder_tokenizer.vocab_file", cfg.encoder_tokenizer.get('vocab_file')),
            )

        encoder_tokenizer, decoder_tokenizer = MTEncDecModel.setup_enc_dec_tokenizers(
            encoder_tokenizer_library=self.encoder_tokenizer_library,
            encoder_tokenizer_model=encoder_tokenizer_model,
            encoder_bpe_dropout=cfg.encoder_tokenizer.get('bpe_dropout', 0.0)
            if cfg.encoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            encoder_r2l=cfg.encoder_tokenizer.get('r2l', False),
            decoder_tokenizer_library=self.decoder_tokenizer_library,
            encoder_tokenizer_vocab_file=encoder_vocab_file,
            decoder_tokenizer_model=decoder_tokenizer_model,
            decoder_bpe_dropout=cfg.decoder_tokenizer.get('bpe_dropout', 0.0)
            if cfg.decoder_tokenizer.get('bpe_dropout', 0.0) is not None
            else 0.0,
            decoder_r2l=cfg.decoder_tokenizer.get('r2l', False),
            encoder_sentencepiece_legacy=cfg.encoder_tokenizer.get('sentencepiece_legacy', False),
            decoder_sentencepiece_legacy=cfg.encoder_tokenizer.get('sentencepiece_legacy', False),
        )
        self.encoder_tokenizer, self.decoder_tokenizer = encoder_tokenizer, decoder_tokenizer

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Set the new vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()
        with open_dict(cfg):
            cfg.ctc_decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        # Override number of classes if placeholder provided
        num_classes = cfg.ctc_decoder["num_classes"]

        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.ctc_decoder["num_classes"] = len(vocabulary)

        self.use_text_data = cfg.get("use_text_data", False)
        self.use_audio_data = cfg.get("use_audio_data", False)
        self.use_transf_encoder = cfg.get("use_transf_encoder", False)

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncDecTransfModelBPE.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncDecTransfModelBPE.from_config_dict(self._cfg.encoder)

        self.tts_model = FastPitchModel.restore_from(self._cfg.tts_model.model_path, map_location="cpu").eval()
        self.enhancer_model = None
        if self._cfg.tts_model.get("enhancer_path", False):
            self.enhancer_model = SpectrogramEnhancerModel.restore_from(
                self._cfg.tts_model.enhancer_path, map_location="cpu"
            ).eval()
        with open(self._cfg.tts_model.speakers_path, "r") as f:
            self.speakers = sorted(map(int, f.read().split()))

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.ctc_decoder or (
                not self._cfg.ctc_decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.ctc_decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.ctc_decoder or not self._cfg.ctc_decoder.feat_in:
                raise ValueError("pctc_aram feat_in of the decoder's config is not set!")

        # CTC decoder
        self.ctc_decoder = EncDecTransfModelBPE.from_config_dict(self._cfg.ctc_decoder)
        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        if self._cfg.encoder['d_model'] != self._cfg.transf_decoder['hidden_size']:
            self.adapter = torch.nn.Linear(self._cfg.encoder['d_model'], self._cfg.transf_decoder['hidden_size'])
        else:
            self.adapter = lambda x: x

        if self.use_transf_encoder:
            # Transformer encoder
            transf_encoder_cfg_dict = OmegaConf.to_container(cfg.get('transf_encoder'))
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
        vocab_size = 8 * ceil(self.tokenizer.vocab_size / 8)
        transf_decoder_cfg_dict = OmegaConf.to_container(cfg.get('transf_decoder'))
        transf_decoder_cfg_dict['vocab_size'] = vocab_size
        library = transf_decoder_cfg_dict.pop('library', 'nemo')
        model_name = transf_decoder_cfg_dict.pop('model_name', None)
        pretrained = transf_decoder_cfg_dict.pop('pretrained', False)
        checkpoint_file = transf_decoder_cfg_dict.pop('checkpoint_file', None)
        self.transf_decoder = get_transformer(
            library=library,
            model_name=model_name,
            pretrained=pretrained,
            config_dict=transf_decoder_cfg_dict,
            encoder=False,
            pre_ln_final_layer_norm=transf_decoder_cfg_dict.get("pre_ln_final_layer_norm", False),
        )

        self.log_softmax = TokenClassifier(
            hidden_size=self.transf_decoder.hidden_size,
            num_classes=vocab_size,
            activation=self._cfg.head.activation,
            log_softmax=self._cfg.head.log_softmax,
            dropout=self._cfg.head.dropout,
            use_transformer_init=self._cfg.head.use_transformer_init,
        )
        self.log_softmax.mlp.layer0.weight = self.transf_decoder.embedding.token_embedding.weight
        std_init_range = 1 / self.transf_decoder.hidden_size ** 0.5
        self.transf_decoder.apply(lambda module: transformer_weights_init(module, std_init_range))
        self.log_softmax.apply(lambda module: transformer_weights_init(module, std_init_range))

        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.transf_decoder.embedding,
            decoder=self.transf_decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.transf_decoder.max_sequence_length,
            beam_size=self._cfg.beam_search.beam_size,
            bos=self.tokenizer.bos_id,
            pad=self.tokenizer.pad_id,
            eos=self.tokenizer.eos_id,
            len_pen=self._cfg.beam_search.len_pen,
            max_delta_length=self._cfg.beam_search.max_generation_delta,
        )
        self.transf_loss = SmoothedCrossEntropyLoss(
            pad_id=self.tokenizer.pad_id, label_smoothing=self._cfg.label_smoothing
        )

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecTransfModelBPE.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.ctc_coef = self._cfg.get("ctc_coef", 0.5)

        self.val_loss = GlobalAverageLossMetric(dist_sync_on_step=False, take_avg_loss=True)

    #         # Setup decoding objects
    #         decoding_cfg = self.cfg.get('decoding', None)

    #         # In case decoding config not found, use default config
    #         if decoding_cfg is None:
    #             decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
    #             with open_dict(self.cfg):
    #                 self.cfg.decoding = decoding_cfg

    #         self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)

    #         # Setup metric objects
    #         self._wer = WERBPE(
    #             decoding=self.decoding,
    #             use_cer=self._cfg.get('use_cer', False),
    #             dist_sync_on_step=True,
    #             log_prediction=self._cfg.get("log_prediction", False),
    #         )

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
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
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

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

                config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'temp_dir': tmpdir}

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    ctc_lp, _, encoded_len, predictions, enc_states, enc_mask = self.forward(
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

                    if return_hypotheses:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                    hypotheses += beam_hypotheses

                    del test_batch
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

    def _text_and_speech_collate_fn(self, batch):
        text_batches, speech_batches = [], []
        for i, b in enumerate(batch):

            if all([isinstance(b_i, tuple) for b_i in b]):
                speech_batches.extend(b)
            elif len(b) == 4:
                speech_batches.append(b)
            else:
                text_batches.append(b)

        speech = _speech_collate_fn(speech_batches, self.tokenizer.pad_id)
        text = bitext_collate_fn(text_batches, self.encoder_tokenizer.pad_id, self.decoder_tokenizer.pad_id,)
        return speech, text

    def setup_training_data(self, train_data_config: Optional[DictConfig]):

        text_config = train_data_config['text']
        audio_config = train_data_config['audio']

        if self.use_text_data:
            # create text dataset
            text_ds = MTEncDecModel._setup_dataset_from_config(
                cfg=text_config,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )

            if self.use_audio_data:
                # create audio-text dataset
                if 'augmentor' in audio_config:
                    augmentor = process_augmentations(audio_config['augmentor'])
                else:
                    augmentor = None
                shuffle_n = 0
                audio_ds = audio_to_text_dataset.get_tarred_dataset(
                    config=audio_config,
                    tokenizer=self.tokenizer,
                    shuffle_n=shuffle_n,
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                    augmentor=augmentor,
                )
                concat_ds = ConcatDataset(
                    datasets=[audio_ds, text_ds],
                    sampling_technique='weighted',
                    sampling_weights=[audio_config['batch_size'], text_config['batch_size']],
                    upsampling_rate=audio_config['upsampling_rate'],
                    global_rank=self.global_rank,
                    world_size=self.world_size,
                )

                # create audio-text data loader
                self._train_dl = DataLoader(
                    dataset=concat_ds,
                    batch_size=audio_config['batch_size'] + text_config['batch_size'],
                    collate_fn=self._text_and_speech_collate_fn,
                    num_workers=train_data_config.get('num_workers', 0),
                    pin_memory=train_data_config.get('pin_memory', False),
                    drop_last=train_data_config.get('drop_last', True),
                )
            else:
                # create text-only data loader
                self._train_dl = MTEncDecModel._setup_dataloader_from_config(cfg=text_config, dataset=text_ds,)
        elif self.use_audio_data:
            # create audio-only data loader
            self._update_dataset_config(dataset_name='train', config=audio_config)
            self._train_dl = self._setup_dataloader_from_config(config=audio_config)

            # Need to set this because if using an IterableDataset, the length of the
            # dataloader is the total number of samples rather than the number of batches,
            # and this messes up the tqdm progress bar. So we set the number of steps manually
            # (to the correct number) to fix this.
            if 'is_tarred' in audio_config and audio_config['is_tarred']:
                # We also need to check if limit_train_batches is already set.
                # If it's an int, we assume that the user has set it to something sane,
                # i.e. <= # training batches, and don't change it. Otherwise, adjust
                # batches accordingly if it's a float (including 1.0).
                if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                    self._trainer.limit_train_batches = int(
                        self._trainer.limit_train_batches
                        * ceil((len(self._train_dl.dataset) / self.world_size) / audio_config['batch_size'])
                    )
                elif self._trainer is None:
                    logging.warning(
                        "Model Trainer was not set before constructing the dataset, incorrect number of "
                        "training batches will be used. Please set the trainer and rebuild the dataset."
                    )

        else:
            raise ValueError("Either text or audio data is required for training.")

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
            "ctc_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "transf_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
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
        # ctc_log_probs = self.ctc_decoder(encoder_output=encoded)
        # greedy_predictions = ctc_log_probs.argmax(dim=-1, keepdim=False)

        enc_states = encoded.permute(0, 2, 1)
        enc_states = self.adapter(enc_states)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        if self.use_transf_encoder:
            enc_states = self.transf_encoder(encoder_states=enc_states, encoder_mask=enc_mask)

        ctc_log_probs = self.ctc_decoder(encoder_output=enc_states.permute(0, 2, 1))
        greedy_predictions = ctc_log_probs.argmax(dim=-1, keepdim=False)

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return ctc_log_probs, transf_log_probs, encoded_len, greedy_predictions, enc_states, enc_mask

    def compute_text_loss(self, batch):

        if batch is None:
            return 0, 0

        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                seq_length = batch[i].shape[-1]
                batch[i] = batch[i].view(-1, seq_length)
        src_ids, src_mask, transcript, tgt_mask, labels = batch
        batch_size = src_ids.shape[0]

        with torch.no_grad():
            speaker_id = random.choice(self.speakers)
            speaker = torch.tensor([speaker_id]).to(src_ids.device)
            signal, signal_len, *_ = self.tts_model(text=src_ids, durs=None, pitch=None, speaker=speaker, pace=1.0)
            if self.enhancer_model is not None:
                signal = self.enhancer_model.forward(input_spectrograms=signal, lengths=signal_len)
            transcript_len = tgt_mask.sum(dim=-1)
            signal = normalize_batch(signal, signal_len, self._cfg.preprocessor["normalize"])[0]
            signal = clean_spectrogram_batch(signal, signal_len)

        ctc_log_probs, transf_log_probs, encoded_len, predictions, enc_states, enc_mask = self.forward(
            processed_signal=signal,
            processed_signal_length=signal_len,
            transcript=transcript,
            transcript_length=transcript_len,
        )

        ctc_loss = 0  # self.ctc_loss(
        #    log_probs=ctc_log_probs,
        #    targets=transcript,
        #    input_lengths=encoded_len,
        #    target_lengths=transcript_len
        # )
        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)
        loss_value = self.ctc_coef * ctc_loss + (1 - self.ctc_coef) * transf_loss

        return loss_value, batch_size

    def compute_audio_loss(self, batch):

        if batch is None:
            return 0, 0

        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]
        batch_size = signal.shape[0]

        ctc_log_probs, transf_log_probs, encoded_len, predictions, enc_states, enc_mask = self.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            transcript=input_ids,
            transcript_length=transcript_len,
        )

        ctc_loss = self.ctc_loss(
            log_probs=ctc_log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)
        loss_value = self.ctc_coef * ctc_loss + (1 - self.ctc_coef) * transf_loss

        return loss_value, batch_size

    # PTL-specific methods
    def training_step(self, batch, batch_nb):

        if self.use_text_data:
            if self.use_audio_data:
                audio_batch, text_batch = batch
            else:
                audio_batch, text_batch = None, batch
        else:
            audio_batch, text_batch = batch, None

        audio_loss, audio_bs = self.compute_audio_loss(audio_batch)
        text_loss, text_bs = self.compute_text_loss(text_batch)
        audio_coef = audio_bs / (audio_bs + text_bs)
        text_coef = text_bs / (audio_bs + text_bs)

        loss_value = audio_coef * audio_loss + text_coef * text_loss

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_loss_audio': audio_loss,
            'train_loss_text': text_loss,
        }

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        #         if (batch_nb + 1) % log_every_n_steps == 0:
        #             self._wer.update(
        #                 predictions=predictions,
        #                 targets=transcript,
        #                 target_lengths=transcript_len,
        #                 predictions_lengths=encoded_len,
        #             )
        #             wer, _, _ = self._wer.compute()
        #             self._wer.reset()
        #             tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            ctc_log_probs, transf_log_probs, encoded_len, predictions, enc_states, enc_mask = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        else:
            ctc_log_probs, transf_log_probs, encoded_len, predictions, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )

        beam_hypotheses = self.beam_search(
            encoder_hidden_states=enc_states, encoder_input_mask=enc_mask, return_beam_scores=False
        )
        loss_value = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        ground_truths = [self.decoder_tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        translations = [self.decoder_tokenizer.ids_to_text(sent) for sent in beam_hypotheses.detach().cpu().tolist()]

        self.val_loss(loss=loss_value, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])

        return {f'{eval_mode}_loss': loss_value, 'translations': translations, 'ground_truths': ground_truths}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, eval_mode="test")

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0, eval_mode: str = "val"):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        for output in outputs:
            eval_loss = getattr(self, 'val_loss').compute()
            translations = list(itertools.chain(*[x['translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))

            # Gather translations and ground truths from all workers
            tr_and_gt = [None for _ in range(self.world_size)]
            # we also need to drop pairs where ground truth is an empty string
            if self.world_size > 1:
                dist.all_gather_object(
                    tr_and_gt, [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']
                )
            else:
                tr_and_gt[0] = [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']

            if self.global_rank == 0:
                _translations = []
                _ground_truths = []
                for rank in range(0, self.world_size):
                    _translations += [t for (t, g) in tr_and_gt[rank]]
                    _ground_truths += [g for (t, g) in tr_and_gt[rank]]

                sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="13a")
                sb_score = sacre_bleu.score * self.world_size

                wer_scores, wer_words = 0, 0
                for h, r in zip(_translations, _ground_truths):
                    wer_words += len(r.split())
                    wer_scores += editdistance.eval(h.split(), r.split())
                wer_score = 1.0 * wer_scores * self.world_size / wer_words

            else:
                sb_score = 0.0
                wer_score = 0.0

            self.log(f"{eval_mode}_loss", eval_loss, sync_dist=True)
            self.log(f"{eval_mode}_sacreBLEU", sb_score, sync_dist=True)
            self.log(f"{eval_mode}_WER", wer_score, sync_dist=True)
            self.val_loss.reset()

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        return self.multi_validation_epoch_end(outputs, dataloader_idx, eval_mode="test")

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
            'labels': self.ctc_decoder.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer
