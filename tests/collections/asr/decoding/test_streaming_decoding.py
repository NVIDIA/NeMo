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

import copy
from typing import Optional

import pytest
import torch
from omegaconf import open_dict
from tqdm.auto import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    BatchedGreedyDecodingState,
    GreedyBatchedLoopLabelsComputerBase,
)
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))

if torch.mps.is_available():
    DEVICES.append(torch.device("mps"))


def load_audio(file_path, target_sr=16000):
    import librosa

    audio, sr = librosa.load(file_path, sr=target_sr)
    return torch.tensor(audio, dtype=torch.float32), sr


def get_model_encoder_output(
    test_audio_filenames,
    num_samples: int,
    model: ASRModel,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    audio_filepaths = test_audio_filenames[:num_samples]

    with torch.no_grad():
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()

        all_inputs, all_lengths = [], []
        for audio_file in tqdm(audio_filepaths, desc="Loading audio files"):
            audio_tensor, _ = load_audio(audio_file)
            all_inputs.append(audio_tensor)
            all_lengths.append(torch.tensor(audio_tensor.shape[0], dtype=torch.int64))

        input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(device=device, dtype=dtype)
        length_batch = torch.tensor(all_lengths, dtype=torch.int64).to(device)

        encoded_outputs, encoded_length = model(input_signal=input_batch, input_signal_length=length_batch)

    return encoded_outputs, encoded_length


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "device,use_cuda_graph_decoder",
    [(device, False) for device in DEVICES] + [(device, True) for device in DEVICES if device.type == "cuda"],
)
@pytest.mark.parametrize("is_tdt", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("max_symbols", [10])
def test_loop_labels_decoding_streaming(
    tmp_path_factory,
    an4_val_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    device: torch.device,
    use_cuda_graph_decoder: bool,
    is_tdt: bool,
    chunk_size: int,
    batch_size: int,
    max_symbols: int,
):
    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large
    model.eval()
    model.to(device=device)

    decoding_cfg = copy.deepcopy(model.cfg.decoding)
    decoding_cfg.strategy = "greedy_batch"
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = use_cuda_graph_decoder
        decoding_cfg.greedy.max_symbols = max_symbols
    model.change_decoding_strategy(decoding_cfg)

    manifest = read_manifest(an4_val_manifest_corrected)
    transcriptions = model.transcribe(audio=str(an4_val_manifest_corrected.absolute()), batch_size=batch_size)
    ref_transcripts = [hyp.text for hyp in transcriptions]

    streaming_transcripts = []
    decoding_computer: GreedyBatchedLoopLabelsComputerBase = model.decoding.decoding._decoding_computer
    with torch.no_grad(), torch.inference_mode():
        for i in range(0, len(manifest), batch_size):
            records = manifest[i : i + batch_size]
            filenames = [record["audio_filepath"] for record in records]
            local_batch_size = len(filenames)
            encoder_output, encoder_output_len = get_model_encoder_output(
                test_audio_filenames=filenames, model=model, num_samples=local_batch_size, device=device
            )
            state: Optional[BatchedGreedyDecodingState] = None
            hyps = None
            encoder_output = encoder_output.transpose(1, 2)
            for t in range(0, encoder_output.shape[1], chunk_size):
                rest_len = encoder_output_len - t
                current_len = torch.full_like(encoder_output_len, fill_value=chunk_size)
                current_len = torch.minimum(current_len, rest_len)
                current_len = torch.maximum(current_len, torch.zeros_like(current_len))
                batched_hyps, _, state = decoding_computer(
                    x=encoder_output[:, t : t + chunk_size],
                    out_len=current_len,
                    prev_batched_state=state,
                )
                new_hyps = batched_hyps_to_hypotheses(batched_hyps, None, batch_size=local_batch_size)
                if hyps is not None:
                    for hyp, new_hyp in zip(hyps, new_hyps):
                        hyp.y_sequence.extend(new_hyp.y_sequence.tolist())
                else:
                    hyps = new_hyps
                    for hyp in hyps:
                        hyp.y_sequence = hyp.y_sequence.tolist()

            for hyp in hyps:
                streaming_transcripts.append(model.tokenizer.ids_to_text(hyp.y_sequence))
    assert ref_transcripts == streaming_transcripts

    model.to(device="cpu")
