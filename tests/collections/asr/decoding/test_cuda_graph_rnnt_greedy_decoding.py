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
import copy
import glob

import jiwer
import pytest
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core.utils.cuda_python_utils import skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported


@pytest.fixture(scope="module")
def stt_en_fastconformer_transducer_xlarge():
    model_name = "stt_en_fastconformer_transducer_xlarge"
    return ASRModel.from_pretrained(model_name, map_location="cpu")


@pytest.fixture(scope="module")
def stt_en_fastconformer_transducer_xxlarge():
    model_name = "stt_en_fastconformer_transducer_xxlarge"
    return ASRModel.from_pretrained(model_name, map_location="cpu")


@pytest.fixture(scope="module")
def stt_en_fastconformer_transducer_large():
    model_name = "stt_en_fastconformer_transducer_large"
    return ASRModel.from_pretrained(model_name, map_location="cpu")


@pytest.mark.with_downloads
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
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
@pytest.mark.parametrize("loop_labels", [False, True])
def test_cuda_graph_rnnt_greedy_decoder(model_name, batch_size, enable_bfloat16, loop_labels: bool, request):
    if not loop_labels:
        skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    device = torch.device("cuda")
    nemo_model = request.getfixturevalue(model_name).to(device)
    decoding_config = copy.deepcopy(nemo_model.cfg.decoding)

    with open_dict(decoding_config):
        decoding_config["greedy"]["max_symbols"] = 5
        decoding_config["greedy"]["loop_labels"] = loop_labels
        decoding_config["greedy"]["use_cuda_graph_decoder"] = False

    nemo_model.change_decoding_strategy(decoding_config)
    audio_filepaths = glob.glob("tests/.data/asr/test/an4/wav/*.wav")

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
        actual_transcripts, _ = nemo_model.transcribe(audio_filepaths, batch_size=batch_size, num_workers=None)

    decoding_config["greedy"]["use_cuda_graph_decoder"] = True

    nemo_model.change_decoding_strategy(decoding_config)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
        fast_transcripts, _ = nemo_model.transcribe(audio_filepaths, batch_size=batch_size, num_workers=None)

    wer = jiwer.wer(actual_transcripts, fast_transcripts)

    assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

    for actual, fast in zip(actual_transcripts, fast_transcripts):
        if actual != fast:
            print("erroneous samples:")
            print("Original transcript:", actual)
            print("New transcript:", fast)


@pytest.mark.with_downloads
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="Test requires 2 GPUs")
@pytest.mark.parametrize("loop_labels", [False, True])
def test_change_devices(loop_labels: bool, stt_en_fastconformer_transducer_xlarge):
    if not loop_labels:
        skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    first_device = torch.device("cuda:0")
    second_device = torch.device("cuda:1")

    batch_size = 8

    nemo_model = stt_en_fastconformer_transducer_xlarge.to(second_device)
    decoding_config = copy.deepcopy(nemo_model.cfg.decoding)

    with open_dict(decoding_config):
        decoding_config["greedy"]["max_symbols"] = 5
        decoding_config["greedy"]["loop_labels"] = loop_labels
        decoding_config["greedy"]["use_cuda_graph_decoder"] = False

    nemo_model.change_decoding_strategy(decoding_config)

    # Test that the model can run successfully when it is first
    # initialized on second_device and then transferred to
    # true_device
    nemo_model.to(first_device)
    audio_filepaths = glob.glob("tests/.data/asr/test/an4/wav/*.wav")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
        second_device_transcripts, _ = nemo_model.transcribe(audio_filepaths, batch_size=batch_size, num_workers=None)

    # Test that the model can run successfully back on second_device
    # after having been first run on first_device. Because the
    # decoder's data structures are lazily initialized, this activates
    # slightly different code than the first case (where the decoder
    # has not run at all), so we want to exercise both cases.
    nemo_model.to(second_device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
        first_device_transcripts, _ = nemo_model.transcribe(audio_filepaths, batch_size=batch_size, num_workers=None)
    # Sanity check: The device we run on should not change execution
    # output.
    assert first_device_transcripts == second_device_transcripts
