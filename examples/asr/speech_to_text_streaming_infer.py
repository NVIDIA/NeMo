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
from argparse import ArgumentParser

import librosa
import torch
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.submodules.conformer_modules import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging


def set_streaming_mode(asr_model, cache_drop_size=0):
    last_channel_num = 0
    last_time_num = 0
    asr_model.encoder.cache_drop_size = cache_drop_size
    for m in asr_model.encoder.layers.modules():
        if hasattr(m, "_max_cache_len"):  # and m._max_cache_len > 0:
            if type(m) == RelPositionMultiHeadAttention:  # or type(m) == RelPositionMultiHeadAttention:
                m._cache_id = last_channel_num
                last_channel_num += 1
                m.cache_drop_size = cache_drop_size

            if type(m) == CausalConv1D:
                m._cache_id = last_time_num
                last_time_num += 1
                m.cache_drop_size = cache_drop_size
    # pre_encoder = asr_model.encoder.pre_encode
    # if type(pre_encoder) == ConvSubsampling:
    #     pre_encoder.is_streaming = True

    return last_channel_num, last_time_num


def model_process(
    asr_model,
    audio_signal,
    length,
    valid_out_len=None,
    cache_last_channel=None,
    cache_last_time=None,
    previous_hypotheses=None,
):

    out = asr_model.encoder(
        audio_signal=audio_signal,
        length=length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
    )
    if len(out) > 2:
        encoded, encoded_len, cache_last_channel_next, cache_last_time_next = out
    else:
        encoded, encoded_len = out
        cache_last_channel_next = cache_last_time_next = None

    if valid_out_len is not None:
        encoded = encoded[:, :, :valid_out_len]
        encoded_len = torch.clamp(encoded_len, max=valid_out_len)

    if hasattr(asr_model, "decoding"):
        best_hyp, _ = asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len.to(encoded.device), return_hypotheses=True, partial_hypotheses=previous_hypotheses
        )
        # greedy_predictions = [hyp.y_sequence for hyp in best_hyp[0]]
        greedy_predictions = best_hyp[0].y_sequence
        # greedy_predictions = []
        # for alignment in best_hyp[0].alignments:
        #     alignment.remove(1024)
        #     greedy_predictions.extend(alignment)
        # greedy_predictions = torch.Tensor(greedy_predictions)

    else:
        log_probs = asr_model.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        best_hyp = None
    return greedy_predictions, cache_last_channel_next, cache_last_time_next, best_hyp


def greedy_merge_ctc(asr_model, preds):
    blank_id = len(asr_model.decoder.vocabulary)
    model_tokenizer = asr_model.tokenizer

    decoded_prediction = []
    previous = blank_id
    for p in preds:
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(int(p))
        previous = p
    hypothesis = model_tokenizer.ids_to_text(decoded_prediction)
    return hypothesis


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to asr model .nemo file",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, help="path to output file", default=None)
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models",
    )

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

    if hasattr(asr_model, "decoding"):
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = True
            # decoding_cfg.greedy.max_symbols = 5
        asr_model.change_decoding_strategy(decoding_cfg)

    encoder_cfg = asr_model.cfg.encoder
    subsampling_factor = encoder_cfg.subsampling_factor
    att_context_style = encoder_cfg.get("att_context_style", "regular")
    conv_context_size = encoder_cfg.get("conv_context_size", [encoder_cfg.conv_kernel_size - 1, 0])
    if att_context_style == "chunked_limited":
        lookahead_steps = encoder_cfg.att_context_size[1] + 1
        cache_drop_size = 0
    else:
        lookahead_steps_att = encoder_cfg.att_context_size[1] * encoder_cfg.n_layers
        lookahead_steps_conv = conv_context_size[1] * encoder_cfg.n_layers
        lookahead_steps = max(lookahead_steps_att, lookahead_steps_conv)
        cache_drop_size = lookahead_steps

    chunk_size = subsampling_factor * (1 + lookahead_steps)
    shift_size = subsampling_factor * ((1 + lookahead_steps) - cache_drop_size)

    asr_model = asr_model.to("cuda")

    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if args.online_normalization:
        model_normalize_type = cfg.preprocessor.normalize
        cfg.preprocessor.normalize = None

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

    if args.online_normalization:
        processed_signal_normalized, x_mean_whole, x_std_whole = normalize_batch(
            x=processed_signal, seq_len=processed_signal_length, normalize_type=model_normalize_type
        )
    else:
        processed_signal_normalized = processed_signal

    #model_normalize_type = {"fixed_mean": x_mean_whole, "fixed_std": x_std_whole}
    #print(x_mean_whole, x_std_whole)

    asr_out_whole, cache_last_channel_next, cache_last_time_next, best_hyp = model_process(
        asr_model=asr_model,
        audio_signal=processed_signal_normalized,
        length=processed_signal_length,
        valid_out_len=None,
        cache_last_channel=None,
        cache_last_time=None,
        previous_hypotheses=None,
    )

    print(asr_out_whole)
    if best_hyp is not None:
        print(best_hyp[0].text)

    #print(greedy_merge_ctc(asr_model, list(asr_out_whole[0].cpu().int().numpy())))

    last_channel_num, last_time_num = set_streaming_mode(asr_model, cache_drop_size=cache_drop_size)
    bs = 1
    last_channel_cache_size = cfg.encoder.att_context_size[0]
    last_time_cache_size = conv_context_size[0]
    cache_last_channel = torch.zeros(
        (last_channel_num, bs, 0, cfg.encoder.d_model), device=asr_model.device, dtype=torch.float32
    )
    cache_last_time = torch.zeros(
        (last_time_num, bs, cfg.encoder.d_model, last_time_cache_size), device=asr_model.device, dtype=torch.float32
    )

    init_chunk_size = 1 + subsampling_factor * lookahead_steps
    init_shift_size = 1
    pre_encode_cache_size = 0  # 8 # 5
    init_cache_pre_encode = torch.zeros(
        (bs, processed_signal.size(-2), pre_encode_cache_size), device=asr_model.device, dtype=torch.float32
    )

    init_audio = processed_signal[:, :, :init_chunk_size]
    if args.online_normalization and init_audio.size(-1) > 1:
        init_audio, x_mean, x_std = normalize_batch(
            x=init_audio, seq_len=torch.tensor([init_audio.size(-1)]), normalize_type=model_normalize_type
        )
        #print(x_mean)
        #print(x_std)
    init_audio = torch.cat((init_cache_pre_encode, init_audio), dim=-1)

    asr_out_stream, cache_last_channel_next, cache_last_time_next, best_hyp = model_process(
        asr_model=asr_model,
        audio_signal=init_audio,
        length=torch.tensor([init_audio.size(-1)]),
        valid_out_len=(init_shift_size - 1) // subsampling_factor + 1,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        previous_hypotheses=None,
    )
    print(asr_out_stream)
    asr_out_stream_total = asr_out_stream

    step_num = 1
    previous_hypotheses = best_hyp
    pre_encode_cache_size = 5
    if type(asr_model.encoder.pre_encode) == ConvSubsampling:
        asr_model.encoder.pre_encode.is_streaming = True

    for i in range(init_shift_size, processed_signal.size(-1), shift_size):
        if i + chunk_size < processed_signal.size(-1):
            valid_out_len = shift_size // subsampling_factor
        else:
            valid_out_len = None

        chunk_audio = processed_signal[:, :, i : i + chunk_size]

        start_pre_encode_cache = i - pre_encode_cache_size
        if start_pre_encode_cache < 0:
            start_pre_encode_cache = 0
        cache_pre_encode = processed_signal[:, :, start_pre_encode_cache:i]
        if cache_pre_encode.size(-1) < pre_encode_cache_size:
            zeros_pads = torch.zeros(
                (bs, chunk_audio.size(-2), pre_encode_cache_size - cache_pre_encode.size(-1)),
                device=asr_model.device,
                dtype=torch.float32,
            )
        else:
            zeros_pads = None

        chunk_audio = torch.cat((cache_pre_encode, chunk_audio), dim=-1)
        if args.online_normalization:
            chunk_audio, x_mean, x_std = normalize_batch(
                x=chunk_audio, seq_len=torch.tensor([chunk_audio.size(-1)]), normalize_type=model_normalize_type
            )
            #print(x_mean)
            #print(x_std)

        if zeros_pads is not None:
            chunk_audio = torch.cat((zeros_pads, chunk_audio), dim=-1)

        (asr_out_stream, cache_last_channel_next, cache_last_time_next, previous_hypotheses,) = model_process(
            asr_model=asr_model,
            audio_signal=chunk_audio,
            length=torch.tensor([chunk_audio.size(-1)], device=asr_model.device),
            valid_out_len=valid_out_len,
            cache_last_channel=cache_last_channel_next,
            cache_last_time=cache_last_time_next,
            previous_hypotheses=previous_hypotheses,
        )
        if last_channel_cache_size >= 0:
            cache_last_channel_next = cache_last_channel_next[:, :, -last_channel_cache_size:, :]
        # print(asr_out_stream)
        print(asr_out_stream.size())
        asr_out_stream_total = torch.cat((asr_out_stream_total, asr_out_stream), dim=-1)
        step_num += 1
        print(processed_signal.size(-1), shift_size, chunk_size, i, len(asr_out_stream_total))
        if i + chunk_size >= processed_signal.size(-1):
            break
    # asr_model = asr_model.to(asr_model.device)
    print(asr_out_stream_total)
    #print(greedy_merge_ctc(asr_model, list(asr_out_stream_total[0].cpu().int().numpy())))

    print(torch.sum(asr_out_stream_total != asr_out_whole))
    print(step_num)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
