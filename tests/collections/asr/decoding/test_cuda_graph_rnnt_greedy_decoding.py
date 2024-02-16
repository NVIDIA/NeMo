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

import glob
import tempfile

import jiwer
import pytest
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core.utils.cuda_python_utils import skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported


@pytest.mark.parametrize(
    ("model_name", "batch_size", "enable_bfloat16"),
    [
        ("stt_en_fastconformer_transducer_xlarge", 8, False),
        ("stt_en_fastconformer_transducer_xxlarge", 8, True),
        pytest.param(
            "stt_en_fastconformer_transducer_large",
            8,
            True,
            marks=pytest.mark.xfail(
                reason="""Cannot instantiate the 
body cuda graph of a conditional node with a persistent kernel (in this case, 
a persistent LSTM), which is triggered in cudnn by using a batch size of 8."""
            ),
        ),
    ],
)
def test_cuda_graph_rnnt_greedy_decoder(model_name, batch_size, enable_bfloat16):
    skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    conf = ASRModel.from_pretrained(model_name, return_config=True)
    with open_dict(conf):
        conf["decoding"]["greedy"]["max_symbols"] = 5
        conf["decoding"]["greedy"]["loop_labels"] = False
        conf["decoding"]["greedy"]["use_cuda_graph_decoder"] = False

    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=conf, f=fp.name)
        nemo_model = ASRModel.from_pretrained(model_name, override_config_path=fp.name, map_location="cuda")

    audio_filepaths = glob.glob("tests/.data/asr/test/an4/wav/*.wav")

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
        actual_transcripts, _ = nemo_model.transcribe(audio_filepaths, batch_size=batch_size, num_workers=None)

    with open_dict(conf):
        conf["decoding"]["greedy"]["use_cuda_graph_decoder"] = True

    nemo_model.change_decoding_strategy(conf["decoding"])

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
        fast_transcripts, _ = nemo_model.transcribe(audio_filepaths, batch_size=batch_size, num_workers=None)

    wer = jiwer.wer(actual_transcripts, fast_transcripts)

    assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

    for actual, fast in zip(actual_transcripts, fast_transcripts):
        if actual != fast:
            print("erroneous samples:")
            print("Original transcript:", actual)
            print("New transcript:", fast)
