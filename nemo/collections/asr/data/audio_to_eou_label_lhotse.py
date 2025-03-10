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

import math
from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeEOUDataset(torch.utils.data.Dataset):
    """
    This dataset is a Lhotse version of diarization dataset in audio_to_diar_label.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define the output types of the dataset."""
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'targets': NeuralType(('B', 'T'), LabelsType()),
            'target_length': NeuralType(tuple('B'), LengthsType()),
            'token_ids': NeuralType(tuple('B', 'T'), LengthsType(), optional=True),
            'token_length': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg, tokenizer: TokenizerSpec, return_cuts: bool = False):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.num_sample_per_mel_frame = int(
            self.cfg.get('window_stride', 0.01) * self.cfg.get('sample_rate', 16000)
        )  # 160 samples for every 1ms by default
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))
        self.eou_token = self.cfg.get('eou_token', '<eou>')

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        targets = []
        tokens = []
        for i in range(len(cuts)):
            targets.append(self.get_frame_labels(audio_lens[i], self.num_sample_per_mel_frame, self.num_mel_frame_per_target_frame))
            tokens.append(torch.as_tensor(self.tokenizer(cuts[i].text + ' ' + self.eou_token)))

        target_lens = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
        targets = collate_vectors(targets, padding_value=0)
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        
        return audio, audio_lens, targets, target_lens, tokens, token_lens
    
    def get_frame_labels(self, num_samples, num_sample_per_mel_frame: int = 160, num_mel_frame_per_asr_frame: int = 8):
        
        mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
        hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)

        targets = torch.ones(hidden_length).long() # speech label
        targets[0] = 2 # start of utterance
        targets[-1] = 3 # end of utterance

        return targets
