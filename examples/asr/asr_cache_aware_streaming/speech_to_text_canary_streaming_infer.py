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
This script can be used to simulate cache-aware streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --audio_file=audio_file.wav \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation.
If compare_vs_offline is not used, then significantly larger batch_size can be used.
Setting `--pad_and_drop_preencoded` would perform the caching for all steps including the first step.
It may result in slightly different outputs from the sub-sampling module compared to offline mode for some techniques like striding and sw_striding.
Enabling it would make it easier to export the model to ONNX.

## Hybrid ASR models
For Hybrid ASR models which have two decoders, you may select the decoder by --set_decoder DECODER_TYPE, where DECODER_TYPE can be "ctc" or "rnnt".
If decoder is not set, then the default decoder would be used which is the RNNT decoder for Hybrid ASR models.

## Multi-lookahead models
For models which support multiple lookaheads, the default is the first one in the list of model.encoder.att_context_size. To change it, you may use --att_context_size, for example --att_context_size [70,1].


## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in offline mode.
But the accuracy would not be very good with small chunks as there is inconsistency between how the model is trained and how the streaming inference is done.
The accuracy of the model on the borders of chunks would not be very good.

To use a model trained with full context, you need to pass the chunk_size and shift_size arguments.
If shift_size is not passed, chunk_size would be used as the shift_size too.
Also argument online_normalization should be enabled to simulate a realistic streaming.
The following command would simulate cache-aware streaming on a pretrained model from NGC with chunk_size of 100, shift_size of 50 and 2 left chunks as left context.
The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.

python speech_to_text_streaming_infer.py \
    --asr_model=stt_en_conformer_ctc_large \
    --chunk_size=100 \
    --shift_size=50 \
    --left_chunks=2 \
    --online_normalization \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

"""


import contextlib
import json
import os
import tempfile
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field, is_dataclass
from typing import Optional

import numpy as np
import torch
from examples.asr.transcribe_speech import TranscriptionConfig
from omegaconf import MISSING, OmegaConf, open_dict
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models.aed_multitask_models import parse_multitask_prompt
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
    ChunkedStreamingAudioBuffer,
)
from nemo.collections.asr.parts.utils.transcribe_utils import (
    compute_output_filename,
    prepare_audio_data,
    restore_transcription_order,
    setup_model,
    write_transcription,
)
from nemo.collections.common.data.utils import move_data_to_device
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class StreamingEvaluationConfig(TranscriptionConfig):

    model_path: Optional[str] = None  # Path to a .nemo file
    model_type: str = (
        "streaming"  # The type of the model to be used for streaming. Could be "streaming" or "offline". The default is "streaming".
    )

    dataset_manifest: Optional[str] = None  # Path to a manifest file containing audio files to perform streaming
    output_path: Optional[str] = None  # path to output file when manifest is used as input
    sort_input_manifest: bool = True  # Whether to sort the input manifest by duration to reduce batched decoding time

    # main deocding params
    batch_size: int = 1  # The batch size to be used to perform streaming in batch mode with multiple streams
    decoding_policy: str = "alignatt"  # streaming decoding policy ["alignatt" or "waitk"]
    waitk_lagging: int = 1  # The frame lagging to be used for waitk decoding policy
    alignatt_thr: int = 4  # The frame threshold to be used for alignatt decoding policy
    xatt_scores_layer: int = -2  # The decoder layer to be used for alignatt decoding policy
    exclude_sink_frames: int = (
        8  # The number of sink frames to be excluded from the xattention scores for alignatt decoding policy
    )
    use_avgpool_for_alignatt: bool = (
        False  # Whether to use avgpool for alignatt decoding policy during most attended frames calculation
    )

    # streaming params
    att_context_size: Optional[list] = (
        None  # Sets the att_context_size for the models which support multiple lookaheads
    )

    # offline params in streaming decoding mode
    window_size: int = (
        14  # The size of encoder embeddings to be used for offline streaming (ms = window_size * subsampling * 10)
    )
    right_context: int = 14  # The right context of encoder embeddings to be used for offline streaming
    use_chunked_features_calculation: bool = (
        True  # Whether to use chunked features calculation for offline models. Should be close to the real streaming scenario.
    )

    pad_and_drop_preencoded: bool = (
        False  # Enables padding the audio input and then dropping the extra steps after the pre-encoding for all the steps including the the first step. It may make the outputs of the downsampling slightly different from offline mode for some techniques like striding or sw_striding.
    )

    use_amp: bool = False  # Whether to use AMP
    device: str = "cuda"  # The device to load the model onto and perform the streaming

    debug_mode: bool = False  # Whether to print more detail in the output


@dataclass
class CanaryData:
    encoded_speech: torch.Tensor = None  # buffer for encoder speech embeddings
    decoder_input_ids: torch.Tensor = None  # tokens ids of initial canary prompt
    tgt: torch.Tensor = None  # buffer with deocoded tokens ids
    decoding_step: int = -1  # current decoding step
    decoder_mems_list: list = None  # decoder caches, helps to reduce the memory usage
    is_last_speech_chunk: torch.Tensor = False  # whether the current chunk is the last speech chunk in the audio
    max_generation_length: int = 512  # maximum number of tokens to be generated for each sample
    max_tokens_per_alignatt_step: int = (
        10  # maximum number of tokens to be generated for each step of alignatt decoding policy (before the last speech chunk)
    )


def compute_laal(delays, source_length, target_length):
    if delays[0] > source_length:
        return delays[0]

    LAAL = 0
    gamma = max(len(delays), target_length) / source_length
    tau = 0
    for t_minus_1, d in enumerate(delays):
        LAAL += d - t_minus_1 / gamma
        tau = t_minus_1 + 1
        if d >= source_length:
            break
    LAAL /= tau

    return LAAL


def compute_alignatt_lagging(batch_samples, predicted_token_ids, canary_data, asr_model, BOW_PREFIX="\u2581"):
    tokens_idx_shift = canary_data.decoder_input_ids.size(-1)
    target_length_word = [len(a['text'].split()) for a in batch_samples]
    laal_list = []
    for i, tokens in enumerate(predicted_token_ids):
        if len(tokens) == 0:
            laal_list.append(5000)
            continue
        audio_encoder_fs = 80
        audio_signal_length = batch_samples[i]["audio_length_ms"]
        # obtain lagging for alignatt
        lagging = []
        for j, cur_t in enumerate(tokens):
            pred_idx = canary_data.tokens_frame_alignment[i, tokens_idx_shift + j] + canary_data.right_context
            cur_t = asr_model.tokenizer.vocab[cur_t]
            eos_token = asr_model.tokenizer.vocab[asr_model.tokenizer.eos_id]
            if (cur_t.startswith(BOW_PREFIX) and cur_t != BOW_PREFIX) or cur_t == eos_token:  # word boundary
                lagging.append(pred_idx * audio_encoder_fs)
            if cur_t == eos_token:
                break
        if len(lagging) == 0:
            lagging.append(0)
        laal = compute_laal(lagging, audio_signal_length, target_length_word[i])
        if torch.is_tensor(laal):
            laal_list.append(laal.item())
        else:
            laal_list.append(laal)
    return laal_list


def compute_waitk_lagging(cfg, batch_samples, predicted_token_ids, canary_data, asr_model, BOW_PREFIX="\u2581"):
    waitk_lagging = cfg.waitk_lagging
    pre_decision_ratio = canary_data.frame_chunk_size
    target_length_word = [len(a['text'].split()) for a in batch_samples]
    laal_list = []
    for i, tokens in enumerate(predicted_token_ids):
        lagging = []
        audio_encoder_fs = 80
        audio_signal_length = batch_samples[i]["audio_length_ms"]
        for j, cur_t in enumerate(tokens):
            cur_src_len = (j + waitk_lagging) * pre_decision_ratio + canary_data.right_context
            cur_src_len *= audio_encoder_fs  # to ms
            cur_src_len = min(audio_signal_length, cur_src_len)
            spm = asr_model.tokenizer.vocab[cur_t]
            # reach word boundary
            if (
                spm.startswith(BOW_PREFIX) and spm != BOW_PREFIX
            ) or cur_t == asr_model.tokenizer.eos_id:  # word boundary
                lagging.append(cur_src_len)
            if cur_t == asr_model.tokenizer.eos_id:
                break
        if len(lagging) == 0:
            lagging.append(0)
        laal = compute_laal(lagging, audio_signal_length, target_length_word[i])
        laal_list.append(laal)
    return laal_list


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def obtain_data_prompt(cfg, asr_model):
    logging.warning(f"Setup lhotse dataloader from {cfg.dataset_manifest}")
    filepaths, sorted_manifest_path = prepare_audio_data(cfg)
    filepaths = sorted_manifest_path if sorted_manifest_path is not None else filepaths

    override_cfg = asr_model.get_transcribe_config()
    override_cfg.batch_size = cfg.batch_size
    override_cfg.num_workers = cfg.num_workers
    override_cfg.return_hypotheses = cfg.return_hypotheses
    override_cfg.channel_selector = cfg.channel_selector
    override_cfg.augmentor = None
    override_cfg.text_field = cfg.gt_text_attr_name
    override_cfg.lang_field = cfg.gt_lang_attr_name
    override_cfg.timestamps = cfg.timestamps
    if hasattr(override_cfg, "prompt"):
        override_cfg.prompt = parse_multitask_prompt(OmegaConf.to_container(cfg.prompt))
    transcribe_cfg = override_cfg

    asr_model._transcribe_on_begin(filepaths, transcribe_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        transcribe_cfg._internal.temp_dir = tmpdir
        dataloader = asr_model._transcribe_input_processing(filepaths, transcribe_cfg)
        batch_example = next(iter(dataloader))
        batch_example = move_data_to_device(batch_example, transcribe_cfg._internal.device)
        return batch_example.prompt


def perform_streaming(cfg, asr_model, streaming_buffer, pad_and_drop_preencoded=False, canary_data=None):
    batch_size = len(streaming_buffer.streams_length)

    if cfg.model_type == "streaming":
        cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
            batch_size=batch_size
        )
    else:
        cache_last_channel, cache_last_time, cache_last_channel_len = None, None, None

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, speech_data in enumerate(streaming_buffer_iter):

        canary_data.step_num = step_num

        if cfg.model_type == "offline" and cfg.use_chunked_features_calculation:
            # compute features for offline model in more fair way (chunk by chunk)
            processed_signal, processed_signal_length = asr_model.preprocessor(
                input_signal=speech_data[2], length=speech_data[3]
            )
            chunk_audio = processed_signal
            chunk_lengths = processed_signal_length
            # TODO replace 160 with the sampling rate from parameters
            canary_data.is_last_speech_chunk = (
                streaming_buffer.streams_length_raw_audio - streaming_buffer.sampling_frames[1] * 160
                <= streaming_buffer.buffer_raw_audio_idx
            )
        else:
            chunk_audio = speech_data[0]
            chunk_lengths = speech_data[1]
            canary_data.is_last_speech_chunk = (
                streaming_buffer.streams_length - streaming_buffer.sampling_frames[1] <= streaming_buffer.buffer_idx
            )

        with torch.inference_mode():
            with autocast:
                # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                # otherwise the last outputs would get dropped
                with torch.no_grad():
                    (
                        pred_out_stream,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                        canary_data,
                    ) = asr_model.conformer_stream_step(
                        cfg=cfg,
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                            asr_model, step_num, pad_and_drop_preencoded
                        ),
                        return_transcription=True,
                        canary_data=canary_data,
                    )

    # get transcription for the last step
    final_streaming_tran = []
    pred_out_stream = []
    for i in range(batch_size):
        transcription_idx = canary_data.tgt[
            i, canary_data.decoder_input_ids.size(-1) : canary_data.current_context_lengths[i]
        ]
        pred_out_stream.append(transcription_idx)
        transcription = asr_model.tokenizer.ids_to_text(transcription_idx.tolist()).strip()
        final_streaming_tran.append(transcription)
        logging.warning(f"[pred_text] {i}: {transcription}")

    return final_streaming_tran, pred_out_stream


@hydra_runner(config_name="StreamingEvaluationConfig", schema=StreamingEvaluationConfig)
def main(cfg: StreamingEvaluationConfig):

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.dataset_manifest is None:
        raise ValueError("dataset_manifest should be non-empty!")

    if cfg.model_path.endswith('.nemo'):
        logging.warning(f"Using local ASR model from {cfg.model_path}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=cfg.model_path)
    else:
        logging.warning(f"Using NGC cloud ASR model {cfg.model_path}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=cfg.model_path)

    # setup att context size
    if cfg.model_type == "streaming" and cfg.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=json.loads(str(cfg.att_context_size)))
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    global autocast
    autocast = torch.amp.autocast(asr_model.device.type, enabled=cfg.use_amp)

    # # configure the decoding config
    multitask_decoding = MultiTaskDecodingConfig()
    multitask_decoding.strategy = "greedy"
    asr_model.change_decoding_strategy(multitask_decoding)

    # setup lhotse dataloader to obtain decoding promt for decoder_input_ids
    decoder_input_ids = obtain_data_prompt(cfg, asr_model)

    asr_model = asr_model.to(cfg.device)
    asr_model.eval()

    online_normalization = False

    if cfg.model_type == "streaming":
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=asr_model,
            online_normalization=online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
    elif cfg.model_type == "offline":
        streaming_buffer = ChunkedStreamingAudioBuffer(
            model=asr_model,
            window_size=cfg.window_size,
            rigtht_context=cfg.right_context,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )

    # stream audio files in a manifest file in batched mode
    samples = []
    all_streaming_tran = []
    all_refs_text = []
    all_answer_text = []
    all_laal = []

    with open(cfg.dataset_manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            samples.append(item)
        # sort the samples by duration to reduce batched decoding time (could be 2 times faster than default random order)
        if cfg.sort_input_manifest:
            samples = sorted(samples, key=lambda x: x['duration'], reverse=True)

    logging.warning(f"Loaded {len(samples)} from the manifest at {cfg.dataset_manifest}.")

    start_time = time.time()
    for sample_idx, sample in tqdm(
        enumerate(samples), desc=f"Eval streaming Canary...", ncols=120, total=len(samples)
    ):
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            sample['audio_filepath'], stream_id=-1
        )
        if "text" in sample:
            all_refs_text.append(sample["text"])
        if "answer" in sample:
            all_answer_text.append(sample["answer"])

        if (sample_idx + 1) % cfg.batch_size == 0 or sample_idx == len(samples) - 1:
            logging.warning("\n=====================================================================================")
            logging.warning(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")

            current_batch_size = len(streaming_buffer.streams_length)

            # initialize the canary data
            canary_data = CanaryData(decoder_input_ids=decoder_input_ids)
            if cfg.model_type == "streaming":
                canary_data.frame_chunk_size = asr_model.encoder.att_context_size[-1] + 1
            else:
                canary_data.frame_chunk_size = cfg.window_size

            canary_data.batch_idxs = torch.arange(current_batch_size, dtype=torch.long, device=asr_model.device)
            canary_data.current_context_lengths = torch.zeros_like(canary_data.batch_idxs) + decoder_input_ids.size(-1)
            canary_data.decoder_input_ids = decoder_input_ids[:current_batch_size]
            canary_data.tgt = torch.full(
                [current_batch_size, canary_data.max_generation_length],
                asr_model.tokenizer.eos,
                dtype=torch.long,
                device=asr_model.device,
            )
            canary_data.tgt[:, : canary_data.decoder_input_ids.size(-1)] = canary_data.decoder_input_ids
            canary_data.tokens_frame_alignment = torch.zeros_like(canary_data.tgt)
            canary_data.active_samples = torch.ones(current_batch_size, dtype=torch.bool, device=asr_model.device)
            canary_data.active_samples_inner_loop = torch.ones(
                current_batch_size, dtype=torch.bool, device=asr_model.device
            )
            canary_data.right_context = cfg.right_context * (1 if cfg.model_type == "offline" else 0)
            canary_data.eos_tokens = torch.full(
                [current_batch_size], asr_model.tokenizer.eos, dtype=torch.long, device=asr_model.device
            )
            canary_data.avgpool2d = torch.nn.AvgPool2d(5, stride=1, padding=2, count_include_pad=False)

            streaming_tran, predicted_token_ids = perform_streaming(
                cfg=cfg,
                asr_model=asr_model,
                streaming_buffer=streaming_buffer,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                canary_data=canary_data,
            )

            all_streaming_tran.extend(streaming_tran)

            # comput decoding latency:
            batch_samples = samples[-current_batch_size:]
            for i in range(streaming_buffer.streams_length.size(-1)):
                batch_samples[i]["audio_length_ms"] = int(streaming_buffer.streams_length[i].item()) * 10

            if cfg.decoding_policy == "waitk":
                laal_list = compute_waitk_lagging(
                    cfg, batch_samples, predicted_token_ids, canary_data, asr_model, BOW_PREFIX="\u2581"
                )
            elif cfg.decoding_policy == "alignatt":
                laal_list = compute_alignatt_lagging(
                    batch_samples, predicted_token_ids, canary_data, asr_model, BOW_PREFIX="\u2581"
                )
            else:
                raise ValueError(f"Unknown decoding policy: {cfg.decoding_policy}")
            all_laal.extend(laal_list)

            streaming_buffer.reset_buffer()

    # write decoded transcriptions and metrics to the output file
    if len(all_refs_text) == len(all_streaming_tran):
        streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
        streaming_laal = int(np.mean(all_laal))
        if all_answer_text:
            streaming_bleu = corpus_bleu(all_streaming_tran, [all_answer_text]).score
            logging.warning(f"BLEU: {round(streaming_bleu, 2)}")
        else:
            logging.warning(f"WER : {round(streaming_wer*100, 2)}%")
        logging.warning(f"LAAL: {streaming_laal}")

    end_time = time.time()
    logging.warning(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

    # stores the results including the transcriptions of the streaming inference in a json file
    if cfg.output_path is not None and len(all_refs_text) == len(all_streaming_tran):

        if cfg.model_type == "streaming":
            chunk_params = f"att-cs{cfg.att_context_size[0]}-{cfg.att_context_size[1]}"
        else:
            chunk_params = f"chunk-w{cfg.window_size}-r{cfg.right_context}"

        fname = (
            "streaming_"
            + os.path.splitext(os.path.basename(cfg.dataset_manifest))[0]
            + "_"
            + chunk_params
            + "_"
            + f"{cfg.decoding_policy}"
            + "_"
            + f"wk-{cfg.waitk_lagging}"
            + "_"
            + f"at_{cfg.alignatt_thr}"
            + ".json"
        )

        hyp_json = os.path.join(cfg.output_path, fname)
        os.makedirs(cfg.output_path, exist_ok=True)
        with open(hyp_json, "w") as out_f:
            for i, hyp in enumerate(all_streaming_tran):
                record = {
                    "text": all_refs_text[i],
                    "pred_text": hyp,
                    "wer": round(word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2),
                    "laal": int(all_laal[i]),
                }
                if all_answer_text:
                    record["answer"] = all_answer_text[i]
                    streaming_bleu_per_file = round(corpus_bleu([hyp], [[all_answer_text[i]]]).score, 2)
                    record["bleu"] = streaming_bleu_per_file
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

        scoring_fname = f"{fname}.scoring.wer"
        scoring_file = os.path.join(cfg.output_path, scoring_fname)
        with open(scoring_file, "w") as out_f:
            if all_answer_text:
                out_f.write(f"Streaming BLEU: {streaming_bleu:.2f}\n")
            else:
                out_f.write(f"Streaming WER : {round(streaming_wer*100, 2)}%\n")
            out_f.write(f"Streaming LAAL: {streaming_laal}\n")


if __name__ == '__main__':
    main()
