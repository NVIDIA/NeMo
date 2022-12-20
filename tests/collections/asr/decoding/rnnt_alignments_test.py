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


# NOTE: the file name does not start with "test_" on purpose to avoid executing
#       these tests outside of the CI machines environment, where test data is
#       stored

import pytest
from examples.asr.transcribe_speech import TranscriptionConfig
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.transcribe_utils import prepare_audio_data, setup_model

TEST_DATA_PATH = "/mnt/datadrive/data/TestData/an4_dataset/an4_val.json"
PRETRAINED_MODEL_NAME = "stt_en_conformer_transducer_small"


@pytest.mark.parametrize("strategy,blank_as_pad", [("greedy", True), ("greedy_batch", True), ("greedy_batch", False),])
def test_rnnt_alignments(strategy: str, blank_as_pad: bool):
    cfg = OmegaConf.structured(TranscriptionConfig(pretrained_name=PRETRAINED_MODEL_NAME))
    cfg.rnnt_decoding.preserve_alignments = True
    cfg.rnnt_decoding.strategy = strategy
    cfg.dataset_manifest = TEST_DATA_PATH
    filepaths = prepare_audio_data(cfg)[0][:10]  # selecting 10 files only

    model = setup_model(cfg, map_location="cuda")[0]
    model.decoder.blank_as_pad = blank_as_pad
    model.change_decoding_strategy(cfg.rnnt_decoding)
    transcriptions = model.transcribe(
        paths2audio_files=filepaths,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        return_hypotheses=True,
        channel_selector=cfg.channel_selector,
    )[0]

    for transcription in transcriptions:
        for align_elem in transcription.alignments:
            for idx, pred in enumerate(align_elem):
                if idx < len(align_elem) - 1:
                    assert pred[1].item() != model.decoder.blank_idx  # all except last have to be non-blank
                else:
                    assert pred[1].item() == model.decoder.blank_idx  # last one has to be blank
