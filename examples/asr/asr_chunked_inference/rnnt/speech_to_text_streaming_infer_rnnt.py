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

"""
Script to perform buffered and streaming inference using RNNT models.

Buffered inference is the primary form of audio transcription when the audio segment is longer than 20-30 seconds.
Also, this is a demonstration of the algorithm that can be used for streaming inference with low latency.
This is especially useful for models such as Conformers, which have quadratic time and memory scaling with
audio duration.

The difference between streaming and buffered inference is the chunk size (or the latency of inference).
Buffered inference will use large chunk sizes (5-10 seconds) + some additional right for context.
Streaming inference will use small chunk sizes (0.1 to 0.25 seconds) + some additional right buffer for context.
Theoretical latency (latency without model inference time) is the sum of the chunk size and the right context.
Keeping large left context (~10s) is not required, but can improve the quality of transcriptions.

Example usage:

```shell
python speech_to_text_streaming_infer_rnnt.py \
    model_path=nvidia/parakeet-rnnt-1.1b \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    right_context_secs=2.0 \
    chunk_secs=2 \
    left_context_secs=10.0 \
    model_stride=8 \
    batch_size=32 \
    clean_groundtruth_text=True \
    langid='en'
```
"""
import copy
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import librosa
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict
from tqdm.auto import tqdm

from nemo.collections.asr.models import EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    GreedyBatchedLoopLabelsComputerBase,
    BatchedGreedyDecodingState,
)
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
from nemo.collections.asr.parts.utils.transcribe_utils import compute_output_filename, setup_model, write_transcription
from nemo.core.config import hydra_runner
from nemo.utils import logging


def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return torch.tensor(audio, dtype=torch.float32), sr


def get_audio_batch(
    test_audio_filenames,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    sample_rate=16000,
):
    audio_filepaths = test_audio_filenames

    with torch.no_grad():
        all_inputs, all_lengths = [], []
        for audio_file in audio_filepaths:
            audio_tensor, _ = load_audio(audio_file, sample_rate=sample_rate)
            all_inputs.append(audio_tensor)
            all_lengths.append(torch.tensor(audio_tensor.shape[0], dtype=torch.int64))

        input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(device=device, dtype=dtype)
        length_batch = torch.tensor(all_lengths, dtype=torch.int64).to(device)

    return input_batch, length_batch


def make_divisible_by(x, factor: int = 8):
    return (x // factor) * factor


@dataclass
class ContextSize:
    left: int
    chunk: int
    right: int

    def total(self) -> int:
        return self.left + self.chunk + self.right

    def subsample(self, factor: int) -> "ContextSize":
        return ContextSize(
            left=self.left // factor,
            chunk=self.chunk // factor,
            right=self.right // factor,
        )


@dataclass
class ContextSizeBatch:
    left: torch.Tensor
    chunk: torch.Tensor
    right: torch.Tensor

    def total(self) -> torch.Tensor:
        return self.left + self.chunk + self.right

    def subsample(self, factor: int) -> "ContextSizeBatch":
        return ContextSizeBatch(
            left=torch.div(self.left, factor, rounding_mode="floor"),
            chunk=torch.div(self.chunk, factor, rounding_mode="floor"),
            right=torch.div(self.right, factor, rounding_mode="floor"),
        )


@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for buffered inference.
    """

    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0
    append_pred: bool = False  # Sets mode of work, if True it will add new field transcriptions.
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()

    # Set to True to output greedy timestamp information (only supported models)
    compute_timestamps: bool = False

    # Set to True to output language ID information
    compute_langs: bool = False

    # Chunked configs
    chunk_secs: float = 1.6  # Chunk length in seconds
    left_context_secs: float = (
        10.0  # left context: larger value improves quality without affecting theoretical latency
    )
    right_context_secs: float = 1.6  # right context

    model_stride: int = (
        8  # Model downsampling factor, 8 for Citrinet and FastConformer models and 4 for Conformer models.
    )

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = True  # allow to select MPS device (Apple Silicon M-series GPU)
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    decoding: RNNTDecodingConfig = field(default_factory=RNNTDecodingConfig)

    # Config for word / character error rate calculation
    calculate_wer: bool = True
    clean_groundtruth_text: bool = False
    langid: str = "en"  # specify this for convert_num_to_words step in groundtruth cleaning
    use_cer: bool = False


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    """
    Transcribes the input audio and can be used to infer long audio files by chunking
    them into smaller segments.
    """
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")  # TODO: param

    cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    filepaths = None
    manifest = cfg.dataset_manifest
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
        manifest = None  # ignore dataset_manifest if audio_dir and dataset_manifest both presents

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            map_location = torch.device('cuda:0')  # use 0th CUDA device
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            map_location = torch.device('mps')
        else:
            map_location = torch.device('cpu')
    else:
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device : {map_location}")

    asr_model, model_name = setup_model(cfg, map_location)

    model_cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    # some changes for streaming scenario
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0

    if model_cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecRNNTBPEModel models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(model_cfg.preprocessor, True)

    # Compute output filename
    cfg = compute_output_filename(cfg, model_name)

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )
        return cfg

    asr_model.freeze()
    asr_model = asr_model.to(asr_model.device)

    # Change Decoding Config
    with open_dict(cfg.decoding):
        cfg.decoding.strategy = "greedy_batch"
        cfg.decoding.greedy.use_cuda_graph_decoder = False  # TODO: fix CUDA graph decoding
        cfg.decoding.preserve_alignments = False
        cfg.decoding.fused_batch_size = -1  # temporarily stop fused batch during inference.
        cfg.decoding.beam.return_best_hypothesis = True  # return and write the best hypothsis only

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if not isinstance(asr_model, EncDecRNNTModel) and not isinstance(asr_model, EncDecHybridRNNTCTCModel):
            raise ValueError("The script supports rnnt model and hybrid model with rnnt decodng!")
        else:
            # rnnt model
            if isinstance(asr_model, EncDecRNNTModel):
                asr_model.change_decoding_strategy(cfg.decoding)

            # hybrid ctc rnnt model with decoder_type = rnnt
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(cfg.decoding, decoder_type='rnnt')

    feature_stride_sec = model_cfg.preprocessor['window_stride']
    features_per_sec = 1.0 / feature_stride_sec
    model_stride = cfg.model_stride
    assert manifest is not None
    records = read_manifest(manifest)
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.preprocessor.featurizer.corrected_pad = True
    asr_model.eval()
    decoding_computer: GreedyBatchedLoopLabelsComputerBase = asr_model.decoding.decoding._decoding_computer
    decoding_computer.disable_cuda_graphs()  # TODO: fix

    audio_sample_rate = model_cfg.preprocessor['sample_rate']

    features_frame2audio_samples = make_divisible_by(int(audio_sample_rate * feature_stride_sec), factor=model_stride)
    encoder_frame2audio_samples = features_frame2audio_samples * model_stride

    context_encoder_frames = ContextSize(
        left=int(cfg.left_context_secs * features_per_sec / model_stride),
        chunk=int(cfg.chunk_secs * features_per_sec / model_stride),
        right=int(cfg.right_context_secs * features_per_sec / model_stride),
    )
    context_samples = ContextSize(
        left=context_encoder_frames.left * model_stride * features_frame2audio_samples,
        chunk=context_encoder_frames.chunk * model_stride * features_frame2audio_samples,
        right=context_encoder_frames.right * model_stride * features_frame2audio_samples,
    )

    full_ctx_audio_samples = context_samples.total()
    logging.info(
        "Corrected contexts (sec): "
        f"Left {context_samples.left / audio_sample_rate:.2f}, "
        f"Chunk {context_samples.chunk / audio_sample_rate:.2f}, "
        f"Right {context_samples.right / audio_sample_rate:.2f}"
    )
    logging.info(
        "Corrected contexts (subsampled encoder frames): "
        f"Left {context_encoder_frames.left}, "
        f"Chunk {context_encoder_frames.chunk}, "
        f"Right {context_encoder_frames.right}"
    )
    logging.info(
        "Corrected contexts (in audio samples): "
        f"Left {context_samples.left}, "
        f"Chunk {context_samples.chunk}, "
        f"Right {context_samples.right}"
    )
    latency_secs = (context_samples.chunk + context_samples.right) / audio_sample_rate
    logging.info(f"Theoretical latency: {latency_secs:.2f} seconds")

    with torch.no_grad(), torch.inference_mode():
        all_hyps = []
        for i in tqdm(range(0, len(records), cfg.batch_size)):
            # get audio
            audio_batch, audio_batch_lengths = get_audio_batch(
                [record["audio_filepath"] for record in records[i : i + cfg.batch_size]],
                device=map_location,
                sample_rate=audio_sample_rate,
            )
            batch_size = audio_batch.shape[0]
            device = audio_batch.device

            # decode audio by chunks

            current_hyps = None
            state: Optional[BatchedGreedyDecodingState] = None
            left_sample = 0
            # right_sample = initial latency in audio samples
            right_sample = min(context_samples.chunk + context_samples.right, audio_batch.shape[1])
            # start with empty buffer
            buffer = torch.zeros([batch_size, 0], dtype=audio_batch.dtype, device=device)
            buffer_size = ContextSize(left=0, chunk=0, right=0)
            buffer_size_batch = ContextSizeBatch(
                left=torch.zeros_like(audio_batch_lengths),
                chunk=torch.zeros_like(audio_batch_lengths),
                right=torch.zeros_like(audio_batch_lengths),
            )
            rest_audio_lengths = audio_batch_lengths.clone()

            # iterate over audio samples
            while left_sample < audio_batch.shape[1]:
                # add samples to buffer
                buffer = torch.cat((buffer, audio_batch[:, left_sample:right_sample]), dim=1)
                added_samples = min(right_sample, audio_batch.shape[1]) - left_sample
                is_last_chunk = right_sample >= audio_batch.shape[1]
                is_last_chunk_batch = added_samples >= rest_audio_lengths
                added_samples_batch = torch.where(is_last_chunk_batch, rest_audio_lengths, added_samples)

                buffer_size.left += buffer_size.chunk
                buffer_size_batch.left += buffer_size_batch.chunk
                buffer_size.chunk = 0
                buffer_size_batch.chunk.fill_(0)
                buffer_size.right += added_samples
                buffer_size_batch.right += added_samples_batch

                if is_last_chunk:
                    buffer_size.chunk = buffer_size.right
                    buffer_size.right = 0
                else:
                    buffer_size.chunk = context_samples.chunk
                    buffer_size.right -= context_samples.chunk
                    assert buffer_size.right == context_samples.right

                buffer_size_batch.chunk = torch.where(
                    is_last_chunk_batch, buffer_size_batch.right, context_samples.chunk
                )
                buffer_size_batch.right = torch.where(
                    is_last_chunk_batch, 0, buffer_size_batch.right - context_samples.chunk
                )

                # fix left context
                buffer_size_batch.left = torch.where(buffer_size_batch.chunk > 0, buffer_size_batch.left, 0)

                # leave only full_ctx_audio_samples in buffer
                extra_samples_in_buffer = max(0, buffer.shape[1] - full_ctx_audio_samples)
                if extra_samples_in_buffer > 0:
                    buffer = buffer[:, extra_samples_in_buffer:]
                    buffer_size.left -= extra_samples_in_buffer
                    buffer_size_batch.left = torch.where(
                        buffer_size_batch.left > extra_samples_in_buffer,
                        buffer_size_batch.left - extra_samples_in_buffer,
                        0,
                    )

                assert buffer_size_batch.total().max().item() == buffer_size.total() == buffer.shape[1]

                encoder_output, encoder_output_len = asr_model(
                    input_signal=buffer,
                    input_signal_length=buffer_size_batch.total(),
                )
                encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
                # ? discard extra encoder output frame
                # remove extra context from encoder_output
                encoder_context = buffer_size.subsample(factor=encoder_frame2audio_samples)
                encoder_context_batch = buffer_size_batch.subsample(factor=encoder_frame2audio_samples)
                # remove left context
                encoder_output = encoder_output[:, encoder_context.left :]

                batched_hyps, _, state = decoding_computer(
                    x=encoder_output,
                    out_len=encoder_context_batch.chunk,  # decode only chunk
                    prev_batched_state=state,
                )
                new_hyps = batched_hyps_to_hypotheses(batched_hyps, None, batch_size=encoder_output.shape[0])
                if current_hyps is not None:
                    for hyp, new_hyp in zip(current_hyps, new_hyps):
                        hyp.y_sequence.extend(new_hyp.y_sequence.tolist())
                else:
                    current_hyps = new_hyps
                    for hyp in current_hyps:
                        hyp.y_sequence = hyp.y_sequence.tolist()

                # move to next sample
                rest_audio_lengths -= added_samples_batch
                left_sample = right_sample
                right_sample = min(right_sample + context_samples.chunk, audio_batch.shape[1])  # add next chunk

            all_hyps.extend(current_hyps)

    for hyp in all_hyps:
        text = asr_model.tokenizer.ids_to_text(hyp.y_sequence)
        hyp.text = text
    # print([hyp.text for hyp in all_hyps])

    output_filename, pred_text_attr_name = write_transcription(
        all_hyps, cfg, model_name, filepaths=filepaths, compute_langs=False, timestamps=False
    )
    logging.info(f"Finished writing predictions to {output_filename}!")

    if cfg.calculate_wer:
        output_manifest_w_wer, total_res, _ = cal_write_wer(
            pred_manifest=output_filename,
            pred_text_attr_name=pred_text_attr_name,
            clean_groundtruth_text=cfg.clean_groundtruth_text,
            langid=cfg.langid,
            use_cer=cfg.use_cer,
            output_filename=None,
        )
        if output_manifest_w_wer:
            logging.info(f"Writing prediction and error rate of each sample to {output_manifest_w_wer}!")
            logging.info(f"{total_res}")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
