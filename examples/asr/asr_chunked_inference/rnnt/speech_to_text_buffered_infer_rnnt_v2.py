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

"""
Script to perform buffered inference using RNNT models.

Buffered inference is the primary form of audio transcription when the audio segment is longer than 20-30 seconds.
This is especially useful for models such as Conformers, which have quadratic time and memory scaling with
audio duration.

The difference between streaming and buffered inference is the chunk size (or the latency of inference).
Buffered inference will use large chunk sizes (5-10 seconds) + some additional buffer for context.
Streaming inference will use small chunk sizes (0.1 to 0.25 seconds) + some additional buffer for context.

# Middle Token merge algorithm

python speech_to_text_buffered_infer_rnnt.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=32 \
    clean_groundtruth_text=True \
    langid='en'

# Longer Common Subsequence (LCS) Merge algorithm

python speech_to_text_buffered_infer_rnnt.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="<remove or path to folder of audio files>" \
    dataset_manifest="<remove or path to manifest>" \
    output_filename="<remove or specify output filename>" \
    total_buffer_in_secs=4.0 \
    chunk_len_in_secs=1.6 \
    model_stride=4 \
    batch_size=32 \
    merge_algo="lcs" \
    lcs_alignment_dir=<OPTIONAL: Some path to store the LCS alignments> 

# NOTE:
    You can use `DEBUG=1 python speech_to_text_buffered_infer_ctc.py ...` to print out the
    predictions of the model, and ground-truth text if presents in manifest.
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

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    GreedyBatchedLoopLabelsComputerBase,
)
from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.asr.parts.utils.rnnt_utils import BatchedGreedyDecodingState, batched_hyps_to_hypotheses
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
    chunk_len_in_secs: float = 1.6  # Chunk length in seconds
    left_context_secs: float = 10.0
    right_context_secs: float = 1.6
    # total_buffer_in_secs: float = 4.0  # Length of buffer (chunk + left and right padding) in seconds
    model_stride: int = (
        8  # Model downsampling factor, 8 for Citrinet and FastConformer models and 4 for Conformer models.
    )

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    decoding: RNNTDecodingConfig = field(default_factory=RNNTDecodingConfig)

    # Decoding configs
    max_steps_per_timestep: int = 5  #'Maximum number of tokens decoded per acoustic timestep'
    stateful_decoding: bool = False  # Whether to perform stateful decoding

    # Merge algorithm for transducers
    merge_algo: Optional[str] = 'middle'  # choices=['middle', 'lcs'], choice of algorithm to apply during inference.
    lcs_alignment_dir: Optional[str] = None  # Path to a directory to store LCS algo alignments

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
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')  # use 0th CUDA device
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        accelerator = 'gpu'
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
    assert manifest is not None
    records = read_manifest(manifest)
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.eval()
    decoding_computer: GreedyBatchedLoopLabelsComputerBase = asr_model.decoding.decoding._decoding_computer
    decoding_computer.disable_cuda_graphs()  # TODO: fix

    audio_sample_rate = model_cfg.preprocessor['sample_rate']

    # def sec_to_spec_frames(seconds: float):
    #     return make_divisible_by(int(seconds / feature_stride_sec), factor=cfg.model_stride)

    frame2audio_samples = make_divisible_by(int(audio_sample_rate * feature_stride_sec), factor=cfg.model_stride)
    left_ctx_audio_samples = int(cfg.left_context_secs * features_per_sec) * frame2audio_samples
    right_ctx_audio_samples = int(cfg.right_context_secs * features_per_sec) * frame2audio_samples
    chunk_ctx_audio_samples = int(cfg.chunk_ctx_secs * features_per_sec) * frame2audio_samples
    full_ctx_audio_samples = left_ctx_audio_samples + chunk_ctx_audio_samples + right_ctx_audio_samples
    logging.info(
        f"Corrected contexts (sec): Left {left_ctx_audio_samples / audio_sample_rate:.2f}, "
        f"Mid {chunk_ctx_audio_samples / audio_sample_rate:.2f}, Right {right_ctx_audio_samples / audio_sample_rate:.2f}"
    )
    latency_secs = (chunk_ctx_audio_samples + right_ctx_audio_samples) / audio_sample_rate
    logging.info(f"Theoretical latency: {latency_secs:.2f} seconds")

    with torch.no_grad(), torch.inference_mode():
        streaming_transcripts = []
        all_hyps = []
        for i in tqdm(range(0, len(records), cfg.batch_size)):
            audio_batch, audio_batch_lengths = get_audio_batch(
                [record["audio_filepath"] for record in records[i : i + cfg.batch_size]],
                device=map_location,
                sample_rate=audio_sample_rate,
            )
            hyps = None
            state: Optional[BatchedGreedyDecodingState] = None

            left_sample = 0
            # right_sample = initial latency in audio samples
            right_sample = chunk_ctx_audio_samples + right_ctx_audio_samples
            # start with empty buffer
            buffer = torch.zeros([audio_batch.shape[0], 0], dtype=audio_batch.dtype, device=audio_batch.device)
            buffer_size = torch.zeros_like(audio_batch_lengths)
            rest_audio_lengths = audio_batch_lengths.clone()
            while left_sample < audio_batch.shape[1]:
                buffer = torch.cat((buffer, audio_batch[:, left_sample:right_sample]), dim=1)
                added_chunk_length = min(right_sample, audio_batch.shape[1]) - left_sample
                current_audio_chunk_len = torch.minimum(
                    rest_audio_lengths, torch.full_like(rest_audio_lengths, fill_value=added_chunk_length)
                )
                buffer_size += current_audio_chunk_len

                # leave only full_ctx_audio_samples in buffer
                if full_ctx_audio_samples > buffer.shape[1]:
                    buffer = buffer[:, -full_ctx_audio_samples:]
                    buffer_size -= full_ctx_audio_samples

                encoder_output, encoder_output_len = asr_model(
                    input_signal=buffer,
                    input_signal_length=buffer_size,
                )
                encoder_output = encoder_output.transpose(1, 2)

                # remove extra context from encoder_output
                # TODO: fix
                # crop_left = int((ctx_start - left) / feature_stride) // audio_sample_rate // cfg.model_stride
                # crop_right = int(right_ctx_audio_frames // audio_sample_rate // cfg.model_stride / feature_stride)
                # encoder_output = encoder_output[:, crop_left : crop_left + chunk_ctx_encoder_frames]
                # encoder_output_len = ...

                batched_hyps, _, state = decoding_computer(
                    x=encoder_output,
                    out_len=encoder_output_len,
                    prev_batched_state=state,
                )
                new_hyps = batched_hyps_to_hypotheses(batched_hyps, None, batch_size=encoder_output.shape[0])
                if hyps is not None:
                    for hyp, new_hyp in zip(hyps, new_hyps):
                        hyp.y_sequence.extend(new_hyp.y_sequence.tolist())
                else:
                    hyps = new_hyps
                    for hyp in hyps:
                        hyp.y_sequence = hyp.y_sequence.tolist()

                # move to next sample
                rest_audio_lengths -= right_sample
                rest_audio_lengths = torch.where(rest_audio_lengths >= 0, rest_audio_lengths, 0)
                left_sample = right_sample
                right_sample += right_ctx_audio_samples  # add next chunk of `right_ctx_audio_samples`

            all_hyps.extend(hyps)
    for hyp in all_hyps:
        text = asr_model.tokenizer.ids_to_text(hyp.y_sequence)
        hyp.text = text
        streaming_transcripts.append(text)
    print(streaming_transcripts)
    # model_stride_in_secs = feature_stride * cfg.model_stride
    # total_buffer = cfg.total_buffer_in_secs
    # chunk_len = float(cfg.chunk_len_in_secs)
    #
    # tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    # mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    # logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")

    # if cfg.merge_algo == 'middle':
    #     frame_asr = BatchedFrameASRRNNT(
    #         asr_model=asr_model,
    #         frame_len=chunk_len,
    #         total_buffer=cfg.total_buffer_in_secs,
    #         batch_size=cfg.batch_size,
    #         max_steps_per_timestep=cfg.max_steps_per_timestep,
    #         stateful_decoding=cfg.stateful_decoding,
    #     )
    #
    # elif cfg.merge_algo == 'lcs':
    #     frame_asr = LongestCommonSubsequenceBatchedFrameASRRNNT(
    #         asr_model=asr_model,
    #         frame_len=chunk_len,
    #         total_buffer=cfg.total_buffer_in_secs,
    #         batch_size=cfg.batch_size,
    #         max_steps_per_timestep=cfg.max_steps_per_timestep,
    #         stateful_decoding=cfg.stateful_decoding,
    #         alignment_basepath=cfg.lcs_alignment_dir,
    #     )
    #     # Set the LCS algorithm delay.
    #     frame_asr.lcs_delay = math.floor(((total_buffer - chunk_len)) / model_stride_in_secs)
    #
    # else:
    #     raise ValueError("Invalid choice of merge algorithm for transducer buffered inference.")
    #
    # hyps = get_buffered_pred_feat_rnnt(
    #     asr=frame_asr,
    #     tokens_per_chunk=tokens_per_chunk,
    #     delay=mid_delay,
    #     model_stride_in_secs=model_stride_in_secs,
    #     batch_size=cfg.batch_size,
    #     manifest=manifest,
    #     filepaths=filepaths,
    #     accelerator=accelerator,
    # )

    output_filename, pred_text_attr_name = write_transcription(
        hyps, cfg, model_name, filepaths=filepaths, compute_langs=False, timestamps=False
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
