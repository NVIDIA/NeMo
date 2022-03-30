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
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

import contextlib
import json
from argparse import ArgumentParser

import onnxruntime
import torch
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import FramewiseStreamingAudioBuffer
from nemo.utils import logging


def extract_transcribtions(hyps):
    if isinstance(hyps[0], Hypothesis):
        transcribtions = []
        for hyp in hyps:
            transcribtions.append(hyp.text)
    else:
        transcribtions = hyps
    return transcribtions


def perform_streaming(asr_model, streaming_buffer, compare_vs_offline=False, debug_mode=False):
    batch_size = len(streaming_buffer.streams_length)
    if compare_vs_offline:
        with autocast():
            processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
            with torch.no_grad():
                (
                    pred_out_offline,
                    transcribed_texts,
                    cache_last_channel_next,
                    cache_last_time_next,
                    best_hyp,
                ) = asr_model.stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    return_transcribtion=True,
                )
        final_offline_tran = extract_transcribtions(transcribed_texts)
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
    else:
        final_offline_tran = None

    cache_last_channel, cache_last_time = asr_model.encoder.get_initial_cache_state(batch_size=batch_size)

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        valid_out_len = streaming_buffer.get_valid_out_len()
        with autocast():
            with torch.no_grad():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    previous_hypotheses,
                ) = asr_model.stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    valid_out_len=valid_out_len,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=True, #True if step_num > 0 else False,
                    return_transcribtion=True,
                )
        if asr_model.encoder.streaming_cfg.last_channel_cache_size >= 0:
            cache_last_channel = cache_last_channel[
                :, :, -asr_model.encoder.streaming_cfg.last_channel_cache_size :, :
            ]

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcribtions(transcribed_texts)}")
        step_num += 1
    final_streaming_tran = extract_transcribtions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    if compare_vs_offline:
        pred_out_stream_cat = torch.cat(pred_out_stream)
        pred_out_offline_cat = torch.cat(pred_out_offline)
        if pred_out_stream_cat.size() == pred_out_offline_cat.size():
            diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )

    return final_streaming_tran, final_offline_tran


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to an ASR model .nemo file",
    )
    parser.add_argument("--onnx_model", type=str, help="Path to the ONNX file of an asr model", default=None)
    parser.add_argument(
        "--device", type=str, help="The device to load the model onto and perform the streaming", default="cuda"
    )
    parser.add_argument("--audio_file", type=str, help="Path to an audio file to perform streaming", default=None)
    parser.add_argument(
        "--manifest_file",
        type=str,
        help="Path to a manifest file containing audio files to perform streaming",
        default=None,
    )
    parser.add_argument("--use_amp", action="store_true", help="Whether to use AMP")
    parser.add_argument("--debug_mode", action="store_true", help="Whether to print more detail in the output.")
    parser.add_argument(
        "--compare_vs_offline",
        action="store_true",
        help="Whether to compare the output of the model with the offline mode.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--online_normalization", default=False, action='store_true', help="Perform normalization on the run."
    )

    args = parser.parse_args()
    if (args.audio_file is None and args.manifest_file is None) or (
        args.audio_file is not None and args.manifest_file is not None
    ):
        raise ValueError("One of the audio_file and manifest_file should be non-empty!")

    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        try:
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=args.asr_model)
        except:
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    # asr_model.encoder.setup_streaming_params(
    #     chunk_size=72*4,
    #     shift_size=72*4,
    #     cache_drop_size=0,
    #     pre_encode_cache_size=8,
    #     valid_out_len=18,
    # )
    asr_model.encoder.setup_streaming_params(
        chunk_size=72,
        shift_size=72,
        cache_drop_size=0,
        pre_encode_cache_size=5,
        valid_out_len=18,
    )
    logging.info(asr_model.encoder.streaming_cfg)
    # asr_model.encoder.setup_streaming_params(
    #     chunk_size=[69, 72],
    #     shift_size=[69, 72],
    #     cache_drop_size=0,
    #     pre_encode_cache_size=[0, 5],
    #     valid_out_len=[18, 18],
    # )

    global autocast
    if (
        args.use_amp
        and torch.cuda.is_available()
        and hasattr(torch.cuda, 'amp')
        and hasattr(torch.cuda.amp, 'autocast')
    ):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # if args.onnx_model is not None:
    #     onnx_model = onnxruntime.InferenceSession(args.onnx_model, providers=['CUDAExecutionProvider'])
    # else:
    #     onnx_model = None

    if hasattr(asr_model, "decoding"):
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = True
            decoding_cfg.greedy.max_symbols = 10
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    streaming_buffer = FramewiseStreamingAudioBuffer(model=asr_model, online_normalization=args.online_normalization)
    if args.audio_file is not None:
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            args.audio_file, stream_id=-1
        )
        # audio_path1 = "/drive3/datasets/data/librispeech_withsp2/LibriSpeech/dev-clean-wav/251-118436-0012.wav"
        # audio_path2 = "/drive3/datasets/data/librispeech_withsp2/LibriSpeech/dev-clean-wav/3081-166546-0019.wav"
        perform_streaming(
            asr_model=asr_model, streaming_buffer=streaming_buffer, compare_vs_offline=args.compare_vs_offline
        )
    else:
        samples = []
        all_streaming_tran = []
        all_offline_tran = []
        all_refs_text = []

        with open(args.manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                samples.append(item)

        logging.info(f"Loaded {len(samples)} from the manifest at {args.manifest_file}.")

        for sample_idx, sample in enumerate(samples):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample['audio_filepath'], stream_id=-1
            )
            all_refs_text.append(sample["text"])
            print(sample["audio_filepath"])
            if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(
                    f"Starting to stream samples from {sample_idx - len(streaming_buffer) + 1} to {sample_idx}..."
                )
                streaming_tran, offline_tran = perform_streaming(
                    asr_model=asr_model,
                    streaming_buffer=streaming_buffer,
                    compare_vs_offline=args.compare_vs_offline,
                    debug_mode=args.debug_mode,
                )
                all_streaming_tran.extend(streaming_tran)
                if args.compare_vs_offline:
                    all_offline_tran.extend(offline_tran)
                streaming_buffer.reset_buffer()
        if args.compare_vs_offline:
            offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text)
            logging.info(f"WER% of offline mode: {round(offline_wer * 100, 2)}")
        streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
        logging.info(f"WER% of streaming mode: {round(streaming_wer*100, 2)}")


if __name__ == '__main__':
    main()
