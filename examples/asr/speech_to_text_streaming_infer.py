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

"""
This script can be used to simulate frame-wise streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in frame-wise streaming mode on a single audio file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --audio_file=audio_file.wav \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

## To evaluate a model in frame-wise streaming mode on a manifest file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation. If compare_vs_offline is not used, then significantly larger batch_size can be used.

"""


import contextlib
import json
import time
from argparse import ArgumentParser

import torch
from omegaconf import open_dict

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


def calc_drop_extra_pre_encoded(asr_model, step_num):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def perform_streaming(asr_model, streaming_buffer, compare_vs_offline=False, debug_mode=False):
    batch_size = len(streaming_buffer.streams_length)
    if compare_vs_offline:
        # would run the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
        # the output of the model in the offline and streaming mode should be exactly the same
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
                    valid_out_len=streaming_buffer.get_valid_out_len(),
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=calc_drop_extra_pre_encoded(asr_model, step_num),
                    return_transcribtion=True,
                )

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcribtions(transcribed_texts)}")

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
        "--online_normalization",
        default=False,
        action='store_true',
        help="Perform normalization on the run per chunk.",
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
        except Exception:
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    logging.info(asr_model.encoder.streaming_cfg)

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

    if hasattr(asr_model, "decoding"):
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = True
            decoding_cfg.greedy.max_symbols = 10
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    if args.online_normalization:
        if asr_model.cfg.normalize in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model does not use normalization, so normalization is skipped."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False
    streaming_buffer = FramewiseStreamingAudioBuffer(model=asr_model, online_normalization=online_normalization)
    if args.audio_file is not None:
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            args.audio_file, stream_id=-1
        )
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

        start_time = time.time()
        for sample_idx, sample in enumerate(samples):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample['audio_filepath'], stream_id=-1
            )
            if "text" in sample:
                all_refs_text.append(sample["text"])
            print(f'Added sample to the buffer: {sample["audio_filepath"]}')

            if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
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

        if args.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
            offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text)
            logging.info(f"WER% of offline mode: {round(offline_wer * 100, 2)}")
        if len(all_refs_text) == len(all_streaming_tran):
            streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
            logging.info(f"WER% of streaming mode: {round(streaming_wer*100, 2)}")
        end_time = time.time()
        logging.info(f"The whole process took: {round(end_time - start_time, 2)}s")


if __name__ == '__main__':
    main()
