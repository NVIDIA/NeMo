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

import random
from typing import Dict, Optional, Tuple
import soundfile

import torch.utils.data
from lhotse.cut import MixedCut, MonoCut, MixTrack, PaddingCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet, SupervisionSegment, MonoCut, Recording, CutSet, AudioSource

import numpy as np

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    speaker_to_target, 
    get_hidden_length_from_sample_length, 
)

class LhotseSpeechToTextSpkBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py. It has the same functionality of LhotseSpeechToTextBpeDataset but also yield speaker target tensor.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'spk_tar_id': NeuralType(('B','T'), LabelsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero',False)
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.fixed_spk_id = self.cfg.get('fixed_spk_id', None)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:

        audio, audio_lens, cuts = self.load_audio(cuts)
        if hasattr('rttm_filepath', cuts[0]) and cuts[0].rttm_filepath:
            spk_targets = [torch.as_tensor(speaker_to_target(cut, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero), dtype=torch.float32) for cut in cuts]
            spk_targets = collate_matrices(spk_targets)
        else:
            spk_targets = None

        tokens = []
        query_cuts = []
        query_speaker_ids = []

        for cut in cuts:
            non_padding_cuts = [track.cut for track in cut.tracks if isinstance(track.cut, MonoCut)]
            if self.fixed_spk_id is None: # Randomly select a speaker during training
                query_spk_id = random.choice(range(len(non_padding_cuts)))
            else: # fix the speaker id for inference
                query_spk_id = self.fixed_spk_id
            query_cut = non_padding_cuts[query_spk_id]
            tokens.append(torch.as_tensor(self.tokenizer(query_cut.custom['text'], cut.supervisions[0].language)))
            query_speaker_ids.append(query_spk_id)
            query_cuts.append(query_cut)
        # query_cuts = CutSet.from_cuts(query_cuts)
        # query, query_lens, _ = self.load_audio(query_cuts)
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        query_speaker_ids = torch.tensor(query_speaker_ids, dtype=torch.long)

        # return audio, audio_lens, query, query_lens, tokens, token_lens, spk_targets, query_speaker_ids # sometimes need query for debugging
        return audio, audio_lens, tokens, token_lens, spk_targets, query_speaker_ids
