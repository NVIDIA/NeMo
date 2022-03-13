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

from argparse import ArgumentParser

import onnxruntime
import torch
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import FramewiseStreamingAudioBuffer
from nemo.utils import logging


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to an ASR model .nemo file",
    )
    parser.add_argument("--onnx_model", type=str, help="Path to the ONNX file of an asr model", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--online_normalization", default=False, action='store_true', help="Perform normalization on the run."
    )

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    if args.onnx_model is not None:
        onnx_model = onnxruntime.InferenceSession(args.onnx_model, providers=['CUDAExecutionProvider'])
    else:
        onnx_model = None

    if hasattr(asr_model, "decoding"):
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = True
            # decoding_cfg.greedy.max_symbols = 5
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to("cuda")
    asr_model.eval()

    audio_path = "/drive3/datasets/data/librispeech_withsp2/LibriSpeech/dev-clean-wav/251-118436-0012.wav"
    audio_path2 = "/drive3/datasets/data/librispeech_withsp2/LibriSpeech/dev-clean-wav/3081-166546-0019.wav"
    streaming_buffer = FramewiseStreamingAudioBuffer(model=asr_model, online_normalization=False)
    processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(audio_path, stream_id=-1)
    processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
        audio_path2, stream_id=-1
    )

    (
        pred_out_offline,
        transcribed_texts,
        cache_last_channel_next,
        cache_last_time_next,
        best_hyp,
    ) = asr_model.stream_step(
        processed_signal=streaming_buffer.buffer,
        processed_signal_length=streaming_buffer.streams_length,
        return_transcribtion=True,
    )
    print(transcribed_texts)
    print(pred_out_offline)

    batch_size = len(streaming_buffer.streams_length)  # args.batch_size
    cache_last_channel, cache_last_time = asr_model.encoder.get_initial_cache_state(batch_size=batch_size)

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        valid_out_len = streaming_buffer.get_valid_out_len()
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
            drop_extra_pre_encoded=True if step_num > 0 else False,
            return_transcribtion=True,
        )
        print(transcribed_texts)

        if asr_model.encoder.streaming_cfg.last_channel_cache_size >= 0:
            cache_last_channel = cache_last_channel[
                :, :, -asr_model.encoder.streaming_cfg.last_channel_cache_size :, :
            ]
        print(pred_out_stream.size())
        step_num += 1
        print(
            processed_signal.size(-1),
            asr_model.encoder.streaming_cfg.shift_size,
            asr_model.encoder.streaming_cfg.chunk_size,
            streaming_buffer.buffer_idx,
            len(pred_out_stream),
        )

    print(pred_out_stream)
    # print(greedy_merge_ctc(asr_model, list(asr_out_stream_total[0].cpu().int().numpy())))

    print(torch.sum(pred_out_stream != pred_out_offline))


if __name__ == '__main__':
    main()
