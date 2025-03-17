# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import math
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig

import torch.utils.data
from lhotse.cut import Cut, CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeEOUDataset(torch.utils.data.Dataset):
    """
    This dataset processes the audio data and the corresponding text data to generate the ASR labels, 
    along with EOU labels for each frame. The audios used in this dataset should only contain speech with 
    NO precedding or following silence. The dataset also randomly pads non-speech frames before and after 
    the audio signal for training EOU prediction task.

    To generate EOU labels, the first frame of the audio will be marked as "start of utterance" (labeled as `2`), 
    while the last frame will be marked as "end of utterance" (labeled as `3`). The rest of the frames in between 
    will be marked as "speech" (labeled as `1`). 
    The padded non-speech signals will be marked as "non-speech" (labeled as 0).
    
    Returns:
        audio: torch.Tensor of audio signal
        audio_lens: torch.Tensor of audio signal length
        eou_targets: torch.Tensor of EOU labels
        eou_target_lens: torch.Tensor of EOU label length
        text_tokens: torch.Tensor of text text_tokens
        text_token_lens: torch.Tensor of text token length

    Padding logic:
    0. Don't pad when `random_padding` is None or during validation/test
    1. randomly draw a probability to decide whether to apply padding
    2. if not padding or audio duration is longer than the maximum duration, 
        1) return the original audio and EOU labels
    3. if apply padding, 
        1) get the max padding duration based on the maximum total duration and the audio duration
        2) randomly draw a total padding duration based on the given distribution
        3) randomly split the total padding duration into pre-padding and post-padding
        4) randomly generate the non-speech signal (audio signal=0) for pre-padding and post-padding
        5) concatenate the pre-padding, audio, and post-padding to get the padded audio signal
        6) update the EOU labels accordingly

    Random padding yaml config:
    ```
    random_padding:
        padding_prob: 0.99  # probability of applying padding
        min_pad_duration: 0.5  # minimum duration of pre/post padding in seconds
        max_total_duration: 30.0  # maximum total duration of the padded audio in seconds
        pad_distribution: 'uniform'  # distribution of padding duration, 'uniform' or 'normal'
        pad_normal_mean: 0.5  # mean of normal distribution for padding duration
        pad_normal_std: 2.0  # standard deviation of normal distribution for padding duration
    ```    
     
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define the output types of the dataset."""
        return {
            'audio': NeuralType(('B', 'T'), AudioSignal()),
            'audio_lens': NeuralType(tuple('B'), LengthsType()),
            'eou_targets': NeuralType(('B', 'T'), LabelsType()),
            'eou_target_lens': NeuralType(tuple('B'), LengthsType()),
            'text_tokens': NeuralType(tuple('B', 'T'), LengthsType(), optional=True),
            'text_token_lens': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg: DictConfig, tokenizer: TokenizerSpec, is_train: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.return_eou_labels = cfg.get('return_eou_labels', True)
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.num_sample_per_mel_frame = int(
            self.cfg.get('window_stride', 0.01) * self.cfg.get('sample_rate', 16000)
        )  # 160 samples for every 1ms by default
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))
        self.eou_token = self.cfg.get('eou_token', '<eou>')
        self.sou_token = self.cfg.get('sou_token', '<sou>')
        self.padding_cfg = self.cfg.get('random_padding', None)

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        audio_signals = []
        audio_lengths = []
        eou_targets = []
        text_tokens = []
        for i in range(len(cuts)):
            eou_targets_i = self.get_frame_labels(cuts[i], audio_lens[i])
            text_tokens_i = self.get_text_tokens(cuts[i])

            audio_i, audio_len_i, eou_targets_i = self.random_pad_audio(
                audio[i], audio_lens[i], eou_targets_i
            )
            audio_signals.append(audio_i)
            audio_lengths.append(audio_len_i)
            eou_targets.append(eou_targets_i)
            text_tokens.append(text_tokens_i)

        audio_signals = collate_vectors(audio_signals, padding_value=0)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
        eou_target_lens = torch.tensor([t.size(0) for t in eou_targets], dtype=torch.long)
        eou_targets = collate_vectors(eou_targets, padding_value=0)
        text_token_lens = torch.tensor([t.size(0) for t in text_tokens], dtype=torch.long)
        text_tokens = collate_vectors(text_tokens, padding_value=0)
        
        if not self.return_eou_labels:
            return audio_signals, audio_lengths, text_tokens, text_token_lens
        return audio_signals, audio_lengths, eou_targets, eou_target_lens, text_tokens, text_token_lens
    
    def _audio_len_to_frame_len(self, num_samples: int):
        """
        Convert the raw audio length to the number of frames after audio encoder.

        self.num_sample_per_mel_frame = int(
            self.cfg.get('window_stride', 0.01) * self.cfg.get('sample_rate', 16000)
        )  # 160 samples for every 1ms by default
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))
        """
        mel_frame_count = math.ceil((num_samples + 1) / self.num_sample_per_mel_frame)
        hidden_length = math.ceil(mel_frame_count / self.num_mel_frame_per_target_frame)
        return hidden_length

    def get_frame_labels(self, cut: Cut, num_samples: int):
        hidden_length = self._audio_len_to_frame_len(num_samples)

        if not cut.has_custom("sou_time") or not cut.has_custom("eou_time"):
            # assume only single speech segment
            eou_targets = torch.ones(hidden_length).long() # speech label
            eou_targets[0] = 2 # start of utterance
            eou_targets[-1] = 3 # end of utterance
            return eou_targets

        sou_time = cut.custom["sou_time"]
        eou_time = cut.custom["eou_time"]

        if not isinstance(sou_time, list) and not isinstance(eou_time, list):
            # only single speech segment
            sou_time = [sou_time]
            eou_time = [eou_time]

        assert len(sou_time) == len(eou_time), f"Number of SOU and EOU do not match: SOU ({len(sou_time)}) vs EOU ({len(eou_time)})"

        eou_targets = torch.zeros(hidden_length).long()
        for i in range(len(sou_time)):
            sou_idx = self._audio_len_to_frame_len(int((sou_time[i] - cut.start) * self.cfg.sample_rate))
            seg_len_in_secs = eou_time[i] - sou_time[i]
            seg_len = self._audio_len_to_frame_len(int(seg_len_in_secs * self.cfg.sample_rate))
            eou_targets[sou_idx:sou_idx+seg_len] = 1
            eou_targets[sou_idx] = 2  # start of utterance
            eou_targets[sou_idx+seg_len-1] = 3 # end of utterance

        return eou_targets

    def get_text_tokens(self, cut: Cut):
        text = cut.text
        if getattr(cut, 'add_sou_eou', True):
            text = f"{self.sou_token} {text} {self.eou_token}"
        return torch.as_tensor(self.tokenizer(text))

    def random_pad_audio(self, audio: torch.Tensor, audio_len: torch.Tensor, eou_targets: torch.Tensor):
        """
        Randomly pad the audio signal with non-speech signal before and after the audio signal.
        Args:
            audio: torch.Tensor of a single audio signal, shape [T]
            audio_len: torch.Tensor of audio signal length, shape [1]
            eou_targets: torch.Tensor of EOU labels, shape [T]
        Returns:
            padded_audio: torch.Tensor of padded audio signal, shape [T+padding]
            padded_audio_len: torch.Tensor of padded audio signal length, shape [1]
            padded_eou_targets: torch.Tensor of padded EOU labels, shape [T+padding]
            padded_eou_targets_len: torch.Tensor of padded EOU label length, shape [1]
        """
        p = np.random.rand()
        if self.padding_cfg is None or not self.is_train or p > self.padding_cfg.padding_prob:
            return audio, audio_len, eou_targets, eou_targets.size(0)
        
        duration = audio_len.item() / self.cfg.sample_rate
        # if already longer than the maximum duration, return the original audio
        if duration >= self.padding_cfg.max_total_duration:
            return audio, audio_len, eou_targets, eou_targets.size(0)

        # apply padding
        audio = audio[:audio_len]
        max_padding_duration = max(0, self.padding_cfg.max_total_duration - duration)
        if max_padding_duration <= self.padding_cfg.min_pad_duration:
            min_padding_duration = 0
        else:
            min_padding_duration = self.padding_cfg.min_pad_duration

        if self.padding_cfg.pad_distribution == 'uniform':
            total_padding_duration = np.random.uniform(min_padding_duration, max_padding_duration)
        elif self.padding_cfg.pad_distribution == 'normal':
            total_padding_duration = np.random.normal(self.padding_cfg.pad_normal_mean, self.padding_cfg.pad_normal_std)
            total_padding_duration = max(min_padding_duration, min(max_padding_duration, total_padding_duration))
        else:
            raise ValueError(f"Unknown padding distribution: {self.padding_cfg.pad_distribution}")
        
        pre_padding_duration = np.random.uniform(0, total_padding_duration)
        post_padding_duration = total_padding_duration - pre_padding_duration

        pre_padding_len = math.ceil(pre_padding_duration * self.cfg.sample_rate)
        post_padding_len = math.ceil(post_padding_duration * self.cfg.sample_rate)

        # pad the audio signal
        pre_padding = torch.zeros(pre_padding_len, dtype=audio.dtype)
        post_padding = torch.zeros(post_padding_len, dtype=audio.dtype)
        padded_audio = torch.cat((pre_padding, audio, post_padding), dim=0)
        padded_audio_len = audio_len + pre_padding_len + post_padding_len

        # pad the EOU labels
        pre_padding_eou_len = self._audio_len_to_frame_len(pre_padding_len)
        post_padding_eou_len = self._audio_len_to_frame_len(post_padding_len)
        pre_padding_eou = torch.zeros(pre_padding_eou_len, dtype=eou_targets.dtype)
        post_padding_eou = torch.zeros(post_padding_eou_len, dtype=eou_targets.dtype)
        padded_eou_targets = torch.cat((pre_padding_eou, eou_targets, post_padding_eou), dim=0)

        return padded_audio, padded_audio_len, padded_eou_targets



