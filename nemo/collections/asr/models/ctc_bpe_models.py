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

from typing import Dict, Optional

import torch

from nemo import logging
from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset
from nemo.collections.asr.metrics.wer_bpe import WordErrorRateBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.common.tokenizers.gpt2_tokenizer import NemoGPT2Tokenizer
from nemo.core.neural_types import *
from nemo.utils.decorators import experimental

__all__ = ['EncDecCTCModelBPE', 'JasperNetBPE', 'QuartzNetBPE']


@experimental
class EncDecCTCModelBPE(EncDecCTCModel):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

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

    def __init__(
        self,
        preprocessor_params: Dict,
        encoder_params: Dict,
        decoder_params: Dict,
        tokenizer_path: str,
        spec_augment_params: Optional[Dict] = None,
    ):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = NemoGPT2Tokenizer(pretrained_model=tokenizer_path)
        logging.info("Tokenizer initialized with {} tokens".format(self.tokenizer.vocab_size))

        # Initialize a dummy vocabulary
        decoder_params['init_params']['vocabulary'] = self.tokenizer.tokenizer.get_vocab()

        # Remove special tokens
        for special_token in self.tokenizer.tokenizer.all_special_tokens:
            if special_token in decoder_params['init_params']['vocabulary']:
                decoder_params['init_params']['vocabulary'].pop(special_token)

        # Override number of classes if placeholder provided
        if decoder_params['init_params']['num_classes'] < 1:

            logging.info("\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                decoder_params['init_params']['num_classes'], self.tokenizer.vocab_size
            ))

            decoder_params['init_params']['num_classes'] = self.tokenizer.vocab_size

        super().__init__(
            preprocessor_params=preprocessor_params,
            encoder_params=encoder_params,
            decoder_params=decoder_params,
            spec_augment_params=spec_augment_params,
        )

        self._wer = WordErrorRateBPE(tokenizer=self.tokenizer, ctc_decode=True)

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        self.train()
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        wer = self._wer(predictions, transcript, transcript_len)
        tensorboard_logs = {
            'train_loss': loss_value,
            'training_wer': wer,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
        }
        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        self.eval()
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        wer = self._wer(predictions, transcript, transcript_len)
        return {'val_loss': loss_value, 'val_wer': wer}


@experimental
class JasperNetBPE(EncDecCTCModelBPE):
    pass


@experimental
class QuartzNetBPE(EncDecCTCModelBPE):
    pass
