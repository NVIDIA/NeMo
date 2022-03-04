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


def to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def model_process(
    asr_model,
    audio_signal,
    length,
    valid_out_len=None,
    cache_last_channel=None,
    cache_last_time=None,
    previous_hypotheses=None,
    onnx_model=None,
):

    if onnx_model is None:
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
            greedy_predictions = best_hyp[0].y_sequence
        else:
            log_probs = asr_model.decoder(encoder_output=encoded)
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
            best_hyp = None
    else:
        ort_inputs = {
            onnx_model.get_inputs()[0].name: to_numpy(audio_signal),
            onnx_model.get_inputs()[1].name: to_numpy(length),
            onnx_model.get_inputs()[2].name: to_numpy(cache_last_channel),
            onnx_model.get_inputs()[3].name: to_numpy(cache_last_time),
        }

        out, cache_last_channel_next, cache_last_time_next = onnx_model.run(None, ort_inputs)
        out = torch.tensor(out)
        cache_last_channel_next = torch.tensor(cache_last_channel_next)
        cache_last_time_next = torch.tensor(cache_last_time_next)
        greedy_predictions = out.argmax(dim=-1, keepdim=False)
        if valid_out_len is not None:
            greedy_predictions = greedy_predictions[:, :valid_out_len]

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
        "--asr_model", type=str, required=True, help="Path to an ASR model .nemo file",
    )
    parser.add_argument("--onnx_model", type=str, help="Path to the ONNX file of an asr model", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--online_normalization", default=False, action='store_true', help="Perform normalization on the run."
    )
    # parser.add_argument("--output_path", type=str, help="path to output file", default=None)

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
    # model_normalize_type = {"fixed_mean": x_mean_whole, "fixed_std": x_std_whole}
    # print(x_mean_whole, x_std_whole)

    audio_path = "/drive3/datasets/data/librispeech_withsp2/LibriSpeech/dev-clean-wav/251-118436-0012.wav"
    streaming_buffer = FramewiseStreamingAudioBuffer(model=asr_model, online_normalization=False)
    processed_signal, processed_signal_length = streaming_buffer.append_audio_file(audio_path)

    # asr_out_whole, cache_last_channel_next, cache_last_time_next, best_hyp = model_process(
    #     asr_model=asr_model,
    #     audio_signal=processed_signal,
    #     length=processed_signal_length,
    #     valid_out_len=None,
    #     cache_last_channel=None,
    #     cache_last_time=None,
    #     previous_hypotheses=None,
    #     onnx_model=None,  # onnx_model,
    # )
    asr_out_whole, cache_last_channel_next, cache_last_time_next, best_hyp = \
        asr_model.stream_step(processed_signal=processed_signal,
                              processed_signal_length=processed_signal_length,
                              valid_out_len=None,
                              cache_last_channel=None,
                              cache_last_time=None,
                              previous_hypotheses=None)


    print(asr_out_whole)
    # if best_hyp is not None:
    #     print(best_hyp[0].text)

    # print(greedy_merge_ctc(asr_model, list(asr_out_whole[0].cpu().int().numpy())))

    # asr_model.encoder.init_streaming_params()
    cache_last_channel, cache_last_time = asr_model.encoder.get_initial_cache_state(batch_size=args.batch_size)

    previous_hypotheses = None
    asr_out_stream_total = None
    streaming_buffer_iter = iter(streaming_buffer)
    for step_num, chunk_audio in enumerate(streaming_buffer_iter):
        if step_num > 0:
            asr_model.encoder.streaming_cfg.drop_extra_pre_encoded = True

        valid_out_len = streaming_buffer.get_valid_out_len()
        # (asr_out_stream, cache_last_channel, cache_last_time, previous_hypotheses,) = model_process(
        #     asr_model=asr_model,
        #     audio_signal=chunk_audio,
        #     length=torch.tensor([chunk_audio.size(-1)], device=asr_model.device),
        #     valid_out_len=valid_out_len,
        #     cache_last_channel=cache_last_channel,
        #     cache_last_time=cache_last_time,
        #     previous_hypotheses=previous_hypotheses,
        #     onnx_model=onnx_model,
        # )

        asr_out_stream, cache_last_channel, cache_last_time, previous_hypotheses \
            = asr_model.stream_step(processed_signal=chunk_audio,
                                    processed_signal_length=torch.tensor([chunk_audio.size(-1)], device=asr_model.device),
                                    valid_out_len=valid_out_len,
                                    cache_last_channel=cache_last_channel,
                                    cache_last_time=cache_last_time,
                                    previous_hypotheses=previous_hypotheses)


        if asr_model.encoder.streaming_cfg.last_channel_cache_size >= 0:
            cache_last_channel = cache_last_channel[:, :, -asr_model.encoder.streaming_cfg.last_channel_cache_size:, :]
        # print(asr_out_stream)
        print(asr_out_stream.size())
        if asr_out_stream_total is None:
            asr_out_stream_total = asr_out_stream
        else:
            asr_out_stream_total = torch.cat((asr_out_stream_total, asr_out_stream), dim=-1)
        step_num += 1
        print(
            processed_signal.size(-1),
            asr_model.encoder.streaming_cfg.shift_size,
            asr_model.encoder.streaming_cfg.chunk_size,
            streaming_buffer.buffer_idx,
            len(asr_out_stream_total),
        )

    print(asr_out_stream_total)
    # print(greedy_merge_ctc(asr_model, list(asr_out_stream_total[0].cpu().int().numpy())))

    print(torch.sum(asr_out_stream_total != asr_out_whole))


if __name__ == '__main__':
    main()
