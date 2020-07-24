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

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from nemo import logging
from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.common import tokenizers
from nemo.core.neural_types import *
from nemo.utils.decorators import experimental

__all__ = ['EncDecCTCModelBPE', 'JasperNetBPE', 'QuartzNetBPE']


@experimental
class EncDecCTCModelBPE(EncDecCTCModel):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

    def __init__(self, cfg: DictConfig, trainer=None):
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        self.tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer, resolve=True)  # type: dict
        self.tokenizer_dir = self.tokenizer_cfg.pop('dir')  # Remove tokenizer directory
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()  # Remove tokenizer_type

        if self.tokenizer_type not in ['bpe', 'wpe']:
            raise ValueError(
                "`tokenizer.type` must be either `bpe` for SentencePiece tokenizer or "
                "`wpe` for BERT based tokenizer"
            )

        if self.tokenizer_type == 'bpe':
            # This is a BPE Tokenizer
            model_path = os.path.join(self.tokenizer_dir, 'tokenizer.model')

            if 'special_tokens' in self.tokenizer_cfg:
                special_tokens = self.tokenizer_cfg['special_tokens']
            else:
                special_tokens = None

            # Update special tokens
            self.tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path, special_tokens=special_tokens)

            vocabulary = {0: '<unk>'}
            with open(os.path.join(self.tokenizer_dir, 'vocab.txt')) as f:
                for i, piece in enumerate(f):
                    piece = piece.replace('\n', '')
                    vocabulary[i + 1] = piece

            # wrapper method to get vocabulary conveniently
            def get_vocab():
                return vocabulary

            # attach utility values to the tokenizer wrapper
            self.tokenizer.tokenizer.vocab_size = len(vocabulary)
            self.tokenizer.tokenizer.get_vocab = get_vocab
            self.tokenizer.tokenizer.all_special_tokens = self.tokenizer.special_token_to_id

        else:
            # This is a WPE Tokenizer
            self.tokenizer_dir = os.path.join(self.tokenizer_dir, 'vocab.txt')
            self.tokenizer = tokenizers.NemoBertTokenizer(vocab_file=self.tokenizer_dir, **self.tokenizer_cfg)

        logging.info(
            "Tokenizer {} initialized with {} tokens".format(
                self.tokenizer.__class__.__name__, self.tokenizer.vocab_size
            )
        )

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Remove special tokens
        for special_token in self.tokenizer.tokenizer.all_special_tokens:
            if special_token in vocabulary:
                vocabulary.pop(special_token)

        # Set the new vocabulary
        cfg.decoder.params.vocabulary = ListConfig(list(vocabulary.values()))

        # Override number of classes if placeholder provided
        if cfg.decoder.params['num_classes'] < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    cfg.decoder.params['num_classes'], self.tokenizer.vocab_size
                )
            )
            cfg.decoder.params['num_classes'] = self.tokenizer.vocab_size

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup metric objects
        self._wer = WERBPE(tokenizer=self.tokenizer, batch_dim_index=0, use_cer=False, ctc_decode=True)

    def transcribe(self, path2audio_file: str) -> str:
        pass

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )

        dataset = AudioToBPEDataset(
            manifest_filepath=config['manifest_filepath'],
            tokenizer=self.tokenizer,
            featurizer=featurizer,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            max_utts=config.get('max_utts', 0),
            trim=config.get('trim_silence', True),
            load_audio=config.get('load_audio', True),
            add_misc=config.get('add_misc', False),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=config['shuffle'],
            num_workers=config.get('num_workers', 0),
        )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), audio_eltype),
            "input_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }


@experimental
class JasperNetBPE(EncDecCTCModelBPE):
    pass


@experimental
class QuartzNetBPE(EncDecCTCModelBPE):
    pass
