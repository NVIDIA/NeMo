# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
import tempfile
import os
from omegaconf import DictConfig
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import DummyManifest

from nemo.collections.asr.data.audio_to_text_lhotse_target_speaker import LhotseSpeechToTextTgtSpkBpeDataset
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory) -> SentencePieceTokenizer:
    tmpdir = tmp_path_factory.mktemp("klingon_tokens")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(map(chr, range(ord('a'), ord('z')))))
    model_path, vocab_path = create_spt_model(
        text_path, vocab_size=32, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir)
    )
    return SentencePieceTokenizer(model_path)


@pytest.mark.unit
def test_lhotse_asr_tgt_spk_dataset(tokenizer):

    cuts = DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1st case: target speaker is query speaker: spk0
        cuts[0].save_audio(os.path.join(tmp_dir, 'query.wav'))
        # add query related info
        cuts[0].speaker_id = 'spk0'
        cuts[0].query_audio_filepath = os.path.join(tmp_dir, 'query.wav')
        cuts[0].query_offset = 0
        cuts[0].query_duration = 1
        cuts[0].query_speaker_id = 'spk0'
        cuts[0].supervisions[0].text = "first"


        # 2nd case: target speaker is not query speaker: spk0 (q) spk1 (t)
        # add query related info
        cuts[1].speaker_id = 'spk1'
        cuts[1].query_audio_filepath = os.path.join(tmp_dir, 'query.wav')
        cuts[1].query_offset = 0
        cuts[1].query_duration = 1
        cuts[1].query_speaker_id = 'spk0'
        cuts[1].supervisions[0].text = ""

        cfg = DictConfig({
            'sample_rate': 16000,
        })


        dataset = LhotseSpeechToTextTgtSpkBpeDataset(tokenizer=tokenizer, cfg=cfg)
        batch = dataset[cuts]

    assert isinstance(batch, tuple)
    assert len(batch) == 5 
    assert all(isinstance(t, torch.Tensor) for t in batch)

    audio, audio_lens, tokens, token_lens, spk_targets = batch

    assert audio.shape == (2, 16000 * 3)
    assert audio_lens.tolist() == [16000 * 3] * 2

    assert tokens.shape == (2, 13)

    assert tokens[0].tolist() == [1, 0, 3, 6, 6, 17, 0, 1, 7, 10, 19, 20, 21]

    assert tokens[1].tolist() == [1, 0, 3, 6, 6, 17, 0, 1, 0, 0, 0, 0, 0]

    assert token_lens.tolist() == [13, 8]

    assert spk_targets.shape == (2, get_hidden_length_from_sample_length(16000 * 3), 4)

    #first case: target speaker is query speaker: spk0
    q_start = 0
    q_end = get_hidden_length_from_sample_length(16000)
    separater_len = get_hidden_length_from_sample_length(16000)
    tgt_start = q_end + separater_len + get_hidden_length_from_sample_length(0)
    tgt_end = tgt_start + get_hidden_length_from_sample_length(16000)
    assert spk_targets[0, q_start:q_end, 0].tolist() == [1.0] * q_end
    assert spk_targets[0, q_end:q_end + separater_len, 0].tolist() == [0.0] * separater_len
    tgt_end = min(tgt_end, spk_targets.shape[1])
    assert spk_targets[0, tgt_start:tgt_end, 0].tolist() == [1.0] * (tgt_end - tgt_start)

    #second case: target speaker is not query speaker: spk0 (q) spk1 (t)
    q_start = 0
    q_end = get_hidden_length_from_sample_length(16000)
    separater_len = get_hidden_length_from_sample_length(16000)
    tgt_start = q_end + separater_len + get_hidden_length_from_sample_length(0)
    tgt_end = tgt_start + get_hidden_length_from_sample_length(16000)
    assert spk_targets[1, q_start:q_end, 0].tolist() == [1.0] * q_end
    assert spk_targets[1, q_end:q_end + separater_len, 0].tolist() == [0.0] * separater_len
    tgt_end = min(tgt_end, spk_targets.shape[1])
    assert spk_targets[1, tgt_start:tgt_end, 1].tolist() == [1.0] * (tgt_end - tgt_start)


