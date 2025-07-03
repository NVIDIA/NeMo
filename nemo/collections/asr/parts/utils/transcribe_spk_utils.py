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
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm
import tracemalloc

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils import manifest_utils, rnnt_utils
from nemo.collections.asr.parts.utils.streaming_tgt_spk_audio_buffer_ctc_batchview_sample_utils import FrameBatchASR_tgt_spk
from nemo.collections.asr.parts.utils.streaming_tgt_spk_audio_buffer_ctc_batchview_dataset_utils import BatchedFrameASRCTC_tgt_spk
from nemo.collections.common.metrics.punct_er import OccurancePunctuationErrorRate
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging, model_utils

from omegaconf import open_dict, OmegaConf

from nemo.collections.asr.parts.utils.transcribe_utils import wrap_transcription, normalize_timestamp_output

def get_buffered_pred_feat_tgt_spk_ctc_batchview_sample(
    asr: Union[FrameBatchASR_tgt_spk],
    frame_len: float,
    tokens_per_chunk: int,
    delay: int,
    preprocessor_cfg: DictConfig,
    model_stride_in_secs: int,
    device: Union[List[int], int],
    manifest: str = None,
    filepaths: List[list] = None,
) -> List[rnnt_utils.Hypothesis]:
    """
    Moved from examples/asr/asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py
    Write all information presented in input manifest to output manifest and removed WER calculation.
    """
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    if filepaths and manifest:
        raise ValueError("Please select either filepaths or manifest")
    if filepaths is None and manifest is None:
        raise ValueError("Either filepaths or manifest shoud not be None")
    if filepaths:
        for l in tqdm(filepaths, desc="Sample:"):
            asr.reset()
            asr.read_audio_file(l, delay, model_stride_in_secs)
            hyp = asr.transcribe(tokens_per_chunk, delay)
            hyps.append(hyp)
    else:
        print(f'Batchview: sample')
        with open(manifest, "r", encoding='utf_8') as mfst_f:
            for l in tqdm(mfst_f, desc="Sample:"):
                asr.reset()
                row = json.loads(l.strip())
                if 'text' in row:
                    refs.append(row['text'])
                audio_file = get_full_path(audio_file=row['audio_filepath'], manifest_file=manifest)
                offset = row['offset']
                duration = row['duration']
                #query info
                query_audio_file = row['query_audio_filepath']
                query_offset = row['query_offset']
                query_duration = row['query_duration']
                #separater info
                separater_freq = asr.asr_model.cfg.test_ds.separater_freq
                separater_duration = asr.asr_model.cfg.test_ds.separater_duration
                separater_unvoice_ratio = asr.asr_model.cfg.test_ds.separater_unvoice_ratio
                # do not support partial audio
                asr.read_audio_file(audio_file, offset, duration, query_audio_file, query_offset, query_duration, separater_freq, separater_duration, separater_unvoice_ratio, delay, model_stride_in_secs)
                hyp = asr.transcribe(tokens_per_chunk, delay)
                hyps.append(hyp)

    if os.environ.get('DEBUG', '0') in ('1', 'y', 't'):
        if len(refs) == 0:
            print("ground-truth text does not present!")
            for hyp in hyps:
                print("hyp:", hyp)
        else:
            for hyp, ref in zip(hyps, refs):
                print("hyp:", hyp)
                print("ref:", ref)

    wrapped_hyps = wrap_transcription(hyps)
    return wrapped_hyps

def get_buffered_pred_feat_tgt_spk_ctc_batchview_dataset(
    asr: BatchedFrameASRCTC_tgt_spk,
    frame_len: int,
    tokens_per_chunk: int,
    delay: int,
    model_stride_in_secs: int,
    batch_size: int,
    manifest: str = None,
    filepaths: List[list] = None,
    accelerator: Optional[str] = 'cpu',
) -> List[rnnt_utils.Hypothesis]:
    """
    Moved from examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py
    Write all information presented in input manifest to output manifest and removed WER calculation.
    """
    hyps = []
    refs = []

    if filepaths and manifest:
        raise ValueError("Please select either filepaths or manifest")
    if filepaths is None and manifest is None:
        raise ValueError("Either filepaths or manifest shoud not be None")

    if manifest:
        filepaths = []
        offsets = []
        durations = []
        query_audio_files = []
        query_offsets = []
        query_durations = []

        #separater info
        separater_freq = asr.asr_model.cfg.test_ds.separater_freq
        separater_duration = asr.asr_model.cfg.test_ds.separater_duration
        separater_unvoice_ratio = asr.asr_model.cfg.test_ds.separater_unvoice_ratio
        print(f'Batchview: dataset')
        with open(manifest, "r", encoding='utf_8') as mfst_f:
            print("Parsing manifest files...")
            for l in mfst_f:
                row = json.loads(l.strip())
                audio_file = get_full_path(audio_file=row['audio_filepath'], manifest_file=manifest)
                offset = row['offset']
                duration = row['duration']
                #query info
                query_audio_file = row['query_audio_filepath']
                query_offset = row['query_offset']
                query_duration = row['query_duration']
                #save to list, each element corresponding to attribute of one sample
                offsets.append(offset)
                durations.append(duration)
                query_audio_files.append(query_audio_file)
                query_offsets.append(query_offset)
                query_durations.append(query_duration)
                
                filepaths.append(audio_file)
                if 'text' in row:
                    refs.append(row['text'])
        assert len(refs) == len(filepaths) == len(offsets) == len(durations) == len(query_audio_files) == len(query_offsets) == len(query_durations)

    with torch.inference_mode():
        with torch.amp.autocast('cpu' if accelerator == 'cpu' else 'cuda'):
            batch = []
            asr.sample_offset = 0
            for idx in tqdm(range(len(filepaths)), desc='Sample:', total=len(filepaths)):
                batch.append((filepaths[idx], offsets[idx], durations[idx], query_audio_files[idx], query_offsets[idx], query_durations[idx])
                )
            
                if len(batch) == batch_size:
                    batch_audio_files = [sample[0] for sample in batch]
                    batch_offsets = [sample[1] for sample in batch]
                    batch_durations = [sample[2] for sample in batch]
                    batch_query_audio_files = [sample[3] for sample in batch]
                    batch_query_offsets = [sample[4] for sample in batch]
                    batch_query_durations= [sample[5] for sample in batch]

                    asr.reset()
                    asr.read_audio_file(batch_audio_files, batch_offsets, batch_durations, batch_query_audio_files, batch_query_offsets, batch_query_durations, separater_freq, separater_duration, separater_unvoice_ratio, delay, model_stride_in_secs)
                    # asr.read_audio_file(batch_audio_files,delay,model_stride_in_secs)
                    hyp_list = asr.transcribe(tokens_per_chunk, delay)
                    hyps.extend(hyp_list)

                    batch.clear()
                    asr.sample_offset += batch_size

            if len(batch) > 0:
                asr.batch_size = len(batch)
                asr.frame_bufferer.batch_size = len(batch)
                batch_audio_files = [sample[0] for sample in batch]
                batch_offsets = [sample[1] for sample in batch]
                batch_durations = [sample[2] for sample in batch]
                batch_query_audio_files = [sample[3] for sample in batch]
                batch_query_offsets = [sample[4] for sample in batch]
                batch_query_durations= [sample[5] for sample in batch]

                asr.reset()
                asr.read_audio_file(batch_audio_files, batch_offsets, batch_durations, batch_query_audio_files, batch_query_offsets, batch_query_durations, separater_freq, separater_duration, separater_unvoice_ratio, delay, model_stride_in_secs)
                hyp_list = asr.transcribe(tokens_per_chunk, delay)
                hyps.extend(hyp_list)

                batch.clear()
                asr.sample_offset += len(batch)

    if os.environ.get('DEBUG', '0') in ('1', 'y', 't'):
        if len(refs) == 0:
            print("ground-truth text does not present!")
            for hyp in hyps:
                print("hyp:", hyp)
        else:
            for hyp, ref in zip(hyps, refs):
                print("hyp:", hyp)
                print("ref:", ref)

    wrapped_hyps = wrap_transcription(hyps)
    return wrapped_hyps

def setup_model(cfg: DictConfig, map_location: torch.device) -> Tuple[ASRModel, str]:
    """ Setup model from cfg and return model and model name for next step """
    if cfg.model_path is not None and cfg.model_path != "None":
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        if cfg.override:
            orig_config = imported_class.restore_from(
                restore_path=cfg.model_path, map_location=map_location,
                return_config=True
            )
            orig_config.rttm_mix_prob = cfg.rttm_mix_prob
            if cfg.rttm_mix_prob == 1:
                orig_config.spk_supervision_strategy = 'rttm'
            elif cfg.rttm_mix_prob == 0:
                orig_config.spk_supervision_strategy = 'diar'
                orig_config.test_ds.spk_tar_all_zero = True
            if cfg.diar_model_path:
                orig_config.diar_model_path = cfg.diar_model_path
            new_config = orig_config
            #set strict to False if model is trained with old diarization model, otherwise set to True
            asr_model = imported_class.restore_from(
                restore_path=cfg.model_path, strict = True, map_location=map_location, override_config_path=new_config
            )
            if cfg.diar_model_path:
                asr_model._init_diar_model()
                asr_model.diarization_model.to(asr_model.device)
            else:
                raise ValueError("Diarization model need to be provided, embedded diarization model loading is not supported yet")
        else:
            asr_model = imported_class.restore_from(
                restore_path=cfg.model_path,map_location=map_location,
            )

        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location,
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    if hasattr(cfg, "model_change") and hasattr(asr_model, "change_attention_model"):
        asr_model.change_attention_model(
            self_attention_model=cfg.model_change.conformer.get("self_attention_model", None),
            att_context_size=cfg.model_change.conformer.get("att_context_size", None),
        )

    return asr_model, model_name



def transcribe_partial_audio(
        
    asr_model, 
    path2manifest: str = None,
    batch_size: int = 4,
    logprobs: bool = False,
    return_hypotheses: bool = False,
    num_workers: int = 0,
    channel_selector: Optional[int] = None,
    augmentor: DictConfig = None,
    decoder_type: Optional[str] = None,
    cfg: DictConfig = None,
) -> List[str]:
    """
    See description of this function in trancribe() in nemo/collections/asr/models/ctc_models.py and nemo/collections/asr/models/rnnt_models.py
    """

    if return_hypotheses and logprobs:
        raise ValueError(
            "Either `return_hypotheses` or `logprobs` can be True at any given time."
            "Returned hypotheses will contain the logprobs."
        )
    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)

    # We will store transcriptions here
    hypotheses = []
    # store spk mapping here
    spk_mappings = []
    # Model's mode and device
    mode = asr_model.training
    device = next(asr_model.parameters()).device
    dither_value = asr_model.preprocessor.featurizer.dither
    pad_to_value = asr_model.preprocessor.featurizer.pad_to
    if decoder_type is not None:  # Hybrid model
        decode_function = (
            asr_model.decoding.rnnt_decoder_predictions_tensor
            if decoder_type == 'rnnt'
            else asr_model.ctc_decoding.ctc_decoder_predictions_tensor
        )
    elif hasattr(asr_model, 'joint'):  # RNNT model
        decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
    else:  # CTC model
        decode_function = asr_model.decoding.ctc_decoder_predictions_tensor

    try:
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        asr_model.eval()
        # Freeze the encoder and decoder modules
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        config = {
            'manifest_filepath': path2manifest,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'channel_selector': channel_selector,
        }
        if augmentor:
            config['augmentor'] = augmentor

        temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
            logits, logits_len, transcript, transcript_len = asr_model.train_val_forward(
                [x.to(device) for x in test_batch], 0
            )

            if isinstance(asr_model, EncDecHybridRNNTCTCModel) and decoder_type == "ctc":
                logits = asr_model.ctc_decoder(encoder_output=logits)

            if logprobs:
                logits = logits.numpy()
                # dump log probs per file
                for idx in range(logits.shape[0]):
                    lg = logits[idx][: logits_len[idx]]
                    hypotheses.append(lg)
            else:
                current_hypotheses =  decode_function(logits, logits_len, return_hypotheses=return_hypotheses,)

                if return_hypotheses:
                    # dump log probs per file
                    for idx in range(logits.shape[0]):
                        current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                        if current_hypotheses[idx].alignments is None:
                            current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence

                hypotheses += current_hypotheses
                
            del logits
            del test_batch

    finally:
        # set mode back to its original value
        asr_model.train(mode=mode)
        asr_model.preprocessor.featurizer.dither = dither_value
        asr_model.preprocessor.featurizer.pad_to = pad_to_value
        if mode is True:
            asr_model.encoder.unfreeze()
            asr_model.decoder.unfreeze()
        logging.set_verbosity(logging_level)

    return hypotheses