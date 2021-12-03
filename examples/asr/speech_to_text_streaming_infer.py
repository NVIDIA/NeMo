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

import copy
import json
import math
import os
from argparse import ArgumentParser

import librosa
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.submodules.conformer_modules import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging

# can_gpu = torch.cuda.is_available()


# def get_wer_feat(mfst, asr, frame_len, tokens_per_chunk, delay, preprocessor_cfg, model_stride_in_secs, device):
#     # Create a preprocessor to convert audio samples into raw features,
#     # Normalization will be done per buffer in frame_bufferer
#     # Do not normalize whatever the model's preprocessor setting is
#     preprocessor_cfg.normalize = "None"
#     preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
#     preprocessor.to(device)
#     hyps = []
#     refs = []
#
#     with open(mfst, "r") as mfst_f:
#         for l in mfst_f:
#             asr.reset()
#             row = json.loads(l.strip())
#             asr.read_audio_file(row['audio_filepath'], delay, model_stride_in_secs)
#             hyp = asr.transcribe(tokens_per_chunk, delay)
#             hyps.append(hyp)
#             refs.append(row['text'])
#
#     wer = word_error_rate(hypotheses=hyps, references=refs)
#     return hyps, refs, wer


def set_streaming_mode(asr_model):
    last_channel_num = 0
    last_time_num = 0
    for m in asr_model.encoder.layers.modules():
        if type(m) == RelPositionMultiHeadAttention:
            m._cache_id = last_channel_num
            last_channel_num += 1
        if type(m) == CausalConv1D:
            m._cache_id = last_time_num
            last_time_num += 1

    return last_channel_num, last_time_num


def model_process(
    asr_model, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_pre_encode=None
):

    out = asr_model.encoder(
        audio_signal=audio_signal,
        length=length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_pre_encode=cache_pre_encode,
    )
    if len(out) == 5:
        encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_pre_encode_next = out
    else:
        encoded, encoded_len = out
        cache_last_channel_next = cache_last_time_next = cache_pre_encode_next = None
    log_probs = asr_model.decoder(encoder_output=encoded)
    greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
    return greedy_predictions, cache_last_channel_next, cache_last_time_next, cache_pre_encode_next


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to asr model .nemo file",
    )
    # parser.add_argument("--test_manifest", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument(
    #     "--total_buffer_in_secs",
    #     type=float,
    #     default=4.0,
    #     help="Length of buffer (chunk + left and right padding) in seconds ",
    # )
    # parser.add_argument("--chunk_len_in_ms", type=int, default=1600, help="Chunk length in milliseconds")
    parser.add_argument("--output_path", type=str, help="path to output file", default=None)
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models",
    )

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    last_channel_num, last_time_num = set_streaming_mode(asr_model)
    asr_model = asr_model.to("cuda")

    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
    preprocessor.to(asr_model.device)

    asr_model.eval()

    audio_path = "/drive3/datasets/data/librispeech_withsp2/LibriSpeech/dev-clean-wav/251-118436-0012.wav"
    audio_sample, _ = librosa.load(audio_path, sr=16000)

    processed_signal, processed_signal_length = preprocessor(
        input_signal=torch.tensor(audio_sample).unsqueeze(0).cuda(), length=torch.tensor([len(audio_sample)]).cuda()
    )

    asr_out_whole, cache_last_channel_next, cache_last_time_next, cache_pre_encode_next = model_process(
        asr_model=asr_model,
        audio_signal=processed_signal,
        length=processed_signal_length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_pre_encode=None,
    )

    print(asr_out_whole)

    # asr_out_whole = asr_model.forward(processed_signal=processed_signal, processed_signal_length=processed_signal_length)

    buffer_size = 4
    init_buffer = 1
    bs = 1

    pre_encode_buffer_size = 5
    last_channel_buffer_size = cfg.encoder.att_context_size[0]
    last_time_buffer_size = cfg.encoder.conv_kernel_size - 1
    cache_pre_encode = torch.zeros(
        (1, bs, pre_encode_buffer_size, processed_signal.size(-2)), device=asr_model.device, dtype=torch.float32
    )
    # cache_last_channel = torch.zeros((last_channel_num, 1, last_channel_buffer_size, cfg.encoder.d_model), device=asr_model.device, dtype=torch.float32)
    cache_last_channel = torch.zeros(
        (last_channel_num, bs, 0, cfg.encoder.d_model), device=asr_model.device, dtype=torch.float32
    )
    cache_last_time = torch.zeros(
        (last_time_num, bs, cfg.encoder.d_model, last_time_buffer_size), device=asr_model.device, dtype=torch.float32
    )

    asr_out_stream, cache_last_channel_next, cache_last_time_next, cache_pre_encode_next = model_process(
        asr_model=asr_model,
        audio_signal=processed_signal[:, :, :init_buffer],
        length=torch.tensor([init_buffer]),
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_pre_encode=cache_pre_encode,
    )
    print(asr_out_stream)
    asr_out_stream_total = asr_out_stream
    for i in range(1, processed_signal.size(-1), buffer_size):
        asr_out_stream, cache_last_channel_next, cache_last_time_next, cache_pre_encode_next = model_process(
            asr_model=asr_model,
            audio_signal=processed_signal[:, :, i:i+buffer_size],
            length=torch.tensor([buffer_size]),
            cache_last_channel=cache_last_channel_next,
            cache_last_time=cache_last_time_next,
            cache_pre_encode=cache_pre_encode_next,
        )
        cache_last_channel_next = cache_last_channel_next[:, :, -last_channel_buffer_size:, :]
        print(asr_out_stream)
        asr_out_stream_total = torch.cat((asr_out_stream_total, asr_out_stream), dim=-1)
    # asr_model = asr_model.to(asr_model.device)
    print(asr_out_stream_total)
    print(torch.sum(asr_out_stream_total != asr_out_whole))
    # with open(args.test_manifest, "r") as mfst_f:
    #     for l in mfst_f:
    #         # asr.reset()
    #         row = json.loads(l.strip())
    #         asr.read_audio_file(row['audio_filepath'], delay, model_stride_in_secs)

    # feature_stride = cfg.preprocessor['window_stride']
    # model_stride_in_secs = feature_stride * args.model_stride
    # total_buffer = args.total_buffer_in_secs
    #
    # chunk_len = args.chunk_len_in_ms / 1000

    # tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    # mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    # print(tokens_per_chunk, mid_delay)
    #
    # frame_asr = FrameBatchASR(
    #     asr_model=asr_model, frame_len=chunk_len, total_buffer=args.total_buffer_in_secs, batch_size=args.batch_size,
    # )
    #
    # hyps, refs, wer = get_wer_feat(
    #     args.test_manifest,
    #     frame_asr,
    #     chunk_len,
    #     tokens_per_chunk,
    #     mid_delay,
    #     cfg.preprocessor,
    #     model_stride_in_secs,
    #     asr_model.device,
    # )
    # logging.info(f"WER is {round(wer, 2)} when decoded with a delay of {round(mid_delay*model_stride_in_secs, 2)}s")

    # if args.output_path is not None:
    #
    #     fname = (
    #         os.path.splitext(os.path.basename(args.asr_model))[0]
    #         + "_"
    #         + os.path.splitext(os.path.basename(args.test_manifest))[0]
    #         + ".json"
    #     )
    #     hyp_json = os.path.join(args.output_path, fname)
    #     os.makedirs(args.output_path, exist_ok=True)
    #     with open(hyp_json, "w") as out_f:
    #         for i, hyp in enumerate(hyps):
    #             record = {
    #                 "pred_text": hyp,
    #                 "text": refs[i],
    #                 "wer": round(word_error_rate(hypotheses=[hyp], references=[refs[i]]) * 100, 2),
    #             }
    #             out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
