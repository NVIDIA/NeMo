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

import json
from dataclasses import dataclass, is_dataclass
from typing import Optional, Union, List, Tuple, Dict, Any

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

from copy import deepcopy
from nemo.collections.asr.parts.utils.diarization_utils import read_seglst, OnlineEvaluation
from nemo.utils import logging

from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.core.config import hydra_runner

from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR, extract_transcriptions
from nemo.collections.asr.parts.utils.speaker_utils import (
audio_rttm_map as get_audio_rttm_map,
rttm_to_labels,
)
from examples.asr.asr_cache_aware_streaming.start_words import COMMON_SENTENCE_STARTS
from nemo.collections.asr.parts.utils.diarization_utils import (
print_sentences,
get_color_palette,
write_txt,
)


from typing import List, Optional
from dataclasses import dataclass
from collections import OrderedDict
import itertools

import time
from functools import wraps
import math

@dataclass
class DiarizationConfig:
    # Required configs
    diar_model_path: Optional[str] = None  # Path to a .nemo file
    diar_pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    # dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    eval_mode: bool = True
    no_der: bool = False
    out_rttm_dir: Optional[str] = None
    opt_style: Optional[str] = None
    
    # General configs
    session_len_sec: float = -1 # End-to-end diarization session length in seconds
    batch_size: int = 1
    num_workers: int = 8
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True # If True, postprocessing will be bypassed
    log: bool = True # If True, log will be printed
    
    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25 # Collar in seconds for DER calculation
    ignore_overlap: bool = False # If True, DER will be calculated only for non-overlapping segments
    
    # Streaming diarization configs
    streaming_mode: bool = True # If True, streaming diarization will be used. 
    mem_len: int = 188
    # mem_refresh_rate: int = 0
    fifo_len: int = 188
    step_len: int = 0
    step_left_context: int = 0
    step_right_context: int = 0

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # ASR Configs
    asr_model: Optional[str] = None
    diar_model: Optional[str] = None
    device: str = 'cuda'
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    use_amp: bool = True
    debug_mode: bool = False
    compare_vs_offline: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = False
    output_path: Optional[str] = None
    pad_and_drop_preencoded: bool = False
    set_decoder: Optional[str] = None # ["ctc", "rnnt"]
    att_context_size: Optional[str] = None
    generate_scripts: bool = True
    
    word_window: int = 50
    fix_speaker_assignments: bool = False
    sentence_break_threshold_in_sec: float = 10000.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    limit_max_spks: int = 2
    print_time: bool = True
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: str = "./"
    ignored_initial_frame_steps: int = 5
    verbose: bool = False

    feat_len_sec: float = 0.01
    finetune_realtime_ratio: float = 0.01
    uppercase_first_letter: bool = True
    remove_pnc: bool = False

def write_txt(path, the_list): 
    outF = open(path, "w")
    for line in the_list:
        outF.write(line)
        outF.write("\n")
    outF.close()

def format_time(seconds):
    minutes = math.floor(seconds / 60)
    sec = seconds % 60
    return f"{minutes}:{sec:05.2f}"
    
def perform_streaming(
    cfg, 
    asr_model, 
    diar_model, 
    streaming_buffer, 
    debug_mode=False):
    batch_size = len(streaming_buffer.streams_length)
    final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    asr_pred_out_stream, diar_pred_out_stream  = None, None
    mem_last_time, fifo_last_time = None, None
    left_offset, right_offset = 0, 0

    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    feat_frame_count = 0
    # write_txt(path="/home/taejinp/projects/mimsasr_sortformer_pr01_fifo_memory/start_flag.txt", the_list=["start"])
    session_start_time = time.time()
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        loop_start_time = time.time()
        with torch.inference_mode():
            with autocast:
                with torch.no_grad(): 
                    (transcribed_speaker_texts,
                    transcribed_texts,
                    asr_pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                    mem_last_time,
                    fifo_last_time,
                    diar_pred_out_stream) = multispk_asr_streamer.perform_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        previous_hypotheses=previous_hypotheses,
                        asr_pred_out_stream=asr_pred_out_stream,
                        diar_pred_out_stream=diar_pred_out_stream,
                        mem_last_time=mem_last_time,
                        fifo_last_time=fifo_last_time,
                        left_offset=left_offset,
                        right_offset=right_offset,
                        pad_and_drop_preencoded=False,
                    )

        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")
        loop_end_time = time.time()
        feat_frame_count += (chunk_audio.shape[-1] - cfg.discarded_frames)
        if cfg.real_time_mode:
            time_diff = max(0, (time.time() - session_start_time) - feat_frame_count * cfg.feat_len_sec)
            eta_min_sec = format_time(time.time() - session_start_time)
            logging.info(f"[   REAL TIME MODE   ] min:sec - {eta_min_sec} "
                         f"Time difference for real-time mode: {time_diff:.4f} seconds")
            time.sleep(max(0, (chunk_audio.shape[-1] - cfg.discarded_frames)*cfg.feat_len_sec - 
                           (loop_end_time - loop_start_time) - time_diff * cfg.finetune_realtime_ratio))
    final_streaming_tran = extract_transcriptions(transcribed_texts)
    return final_streaming_tran, final_offline_tran


@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)
        
    if cfg.diar_model_path is None and cfg.diar_pretrained_name is None:
        raise ValueError("Both cfg.diar_model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.manifest_file is None:
        raise ValueError("Both cfg.audio_dir and cfg.manifest_file cannot be None!")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    if cfg.diar_model_path.endswith(".ckpt"):
        diar_model = SortformerEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.diar_model_path, 
                                                                  map_location=map_location, strict=False)
    elif cfg.diar_model_path.endswith(".nemo"):
        diar_model = SortformerEncLabelModel.restore_from(restore_path=cfg.diar_model_path, 
                                                          map_location=map_location)
    else:
        raise ValueError("cfg.diar_model_path must end with.ckpt or.nemo!")
    
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)
    
    diar_model = diar_model.eval()
    diar_model._cfg.test_ds.manifest_filepath = cfg.manifest_file
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    
    # Model setup for inference 
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)    
    
    # Steaming mode setup 
    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_modules.step_len = cfg.step_len
    diar_model.sortformer_modules.mem_len = cfg.mem_len
    diar_model.sortformer_modules.step_left_context = cfg.step_left_context
    diar_model.sortformer_modules.step_right_context = cfg.step_right_context
    diar_model.sortformer_modules.fifo_len = cfg.fifo_len
    diar_model.sortformer_modules.log = cfg.log

    args = cfg
    if (args.audio_file is None and args.manifest_file is None) or (
        args.audio_file is not None and args.manifest_file is not None
    ):
        raise ValueError("One of the audio_file and manifest_file should be non-empty!")

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)

    logging.info(asr_model.encoder.streaming_cfg)
    if args.set_decoder is not None:
        if hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=args.set_decoder)
        else:
            raise ValueError("Decoder cannot get changed for non-Hybrid ASR models.")

    if args.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=json.loads(args.att_context_size))
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    global autocast
    autocast = torch.amp.autocast(asr_model.device.type, enabled=args.use_amp)

    # configure the decoding config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    # chunk_size is set automatically for models trained for streaming. 
    # For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if args.chunk_size > 0:
        if args.shift_size < 0:
            shift_size = args.chunk_size
        else:
            shift_size = args.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
        )

    # In streaming, offline normalization is not feasible as we don't have access to the 
    # whole audio at the beginning When online_normalization is enabled, the normalization 
    # of the input features (mel-spectrograms) are done per step It is suggested to train 
    # the streaming models without any normalization in the input features.
    if args.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has"
                "no normalization in the feature extration part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
    )
    
    if args.audio_file is not None:
        # stream a single audio file
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            args.audio_file, stream_id=-1
        )
        streaming_tran, offline_tran = perform_streaming(
            cfg=cfg,
            asr_model=asr_model,
            diar_model=diar_model,
            streaming_buffer=streaming_buffer,
        )
    else:
        # stream audio files in a manifest file in batched mode
        samples = []
        all_refs_text = []

        with open(args.manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                samples.append(item)

        # Override batch size: The batch size should be equal to the number of samples in the manifest file
        args.batch_size = len(samples)
        logging.info(f"Loaded {len(samples)} from the manifest at {args.manifest_file}.")

        start_time = time.time()
        for sample_idx, sample in enumerate(samples):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample['audio_filepath'], offset=sample['offset'], duration=sample['duration'], stream_id=-1
            )
            if "text" in sample:
                all_refs_text.append(sample["text"])
            logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

            if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                streaming_tran, offline_tran = perform_streaming(
                    cfg=cfg,
                    asr_model=asr_model,
                    diar_model=diar_model,
                    streaming_buffer=streaming_buffer,
                )
                
if __name__ == '__main__':
    main()