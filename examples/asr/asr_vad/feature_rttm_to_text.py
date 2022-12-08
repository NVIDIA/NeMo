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

import contextlib
import json
import os
import time
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from nemo.collections.asr.data import feature_to_text_dataset
from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.metrics.wer import CTCDecodingConfig, word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


@dataclass
class TranscriptionConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    use_pure_noise: bool = False  # whether input is pure noise or not
    use_rttm: bool = True  # whether to use RTTM
    use_feature: bool = True  # whether to use preprocessed audio features
    normalize: Optional[str] = "post_norm"  # choices=[pre_norm, post_norm]
    frame_unit_time_secs: float = 0.01  # unit time per frame in seconds, equal to `window_stride` in ASR configs
    profiling: bool = False  # whether to enable pytorch profiling

    # General configs
    batch_size: int = 1
    num_workers: int = 8
    output_filename: Optional[str] = None  # will be automatically set by the program
    pred_name_postfix: Optional[str] = None  # If you need to use another model name, rather than standard one.

    # Set to True to output language ID information
    compute_langs: bool = False

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = CTCDecodingConfig()

    # Decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig(fused_batch_size=-1)


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.dataset_manifest is None:
        raise ValueError("cfg.dataset_manifest cannot be None!")

    # setup profiling, note that profiling will significantly increast the total runtime
    if cfg.profiling:
        logging.info("Profiling enabled")
        profile_fn = profile
        record_fn = record_function
    else:
        logging.info("Profiling disabled")

        @contextlib.contextmanager
        def profile_fn(*args, **kwargs):
            yield

        @contextlib.contextmanager
        def record_fn(*args, **kwargs):
            yield

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(
            restore_path=cfg.model_path, map_location=map_location
        )  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    asr_model = asr_model.eval()

    # set True to collect additional transcription information
    return_hypotheses = False

    # Setup decoding strategy
    is_rnnt = False
    if hasattr(asr_model, 'change_decoding_strategy'):
        # Check if ctc or rnnt model
        if hasattr(asr_model, 'joint'):  # RNNT model
            is_rnnt = True
            cfg.rnnt_decoding.fused_batch_size = -1
            cfg.rnnt_decoding.compute_langs = cfg.compute_langs
            asr_model.change_decoding_strategy(cfg.rnnt_decoding)
            decode_function = asr_model.decoding.rnnt_decoder_predictions_tensor
        else:
            asr_model.change_decoding_strategy(cfg.ctc_decoding)
            decode_function = asr_model.decoding.ctc_decoder_predictions_tensor

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        if cfg.pred_name_postfix is not None:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{cfg.pred_name_postfix}.json')
        else:
            tag = f"_{cfg.normalize}"
            if cfg.use_rttm:
                tag += "_rttm"
            if not cfg.use_feature:
                tag += "_audio"
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'{tag}_{model_name}.json')

    # Setup dataloader
    if cfg.use_feature:
        logging.info("Using preprocessed audio features as input...")
        data_config = {
            "manifest_filepath": cfg.dataset_manifest,
            "normalize": cfg.normalize,
            "frame_unit_time_secs": cfg.frame_unit_time_secs,
            "use_rttm": cfg.use_rttm,
        }
        logging.info(f"use_rttm={cfg.use_rttm}")
        if hasattr(asr_model, "tokenizer"):
            dataset = feature_to_text_dataset.get_bpe_dataset(config=data_config, tokenizer=asr_model.tokenizer)
        else:
            data_config["labels"] = asr_model.decoder.vocabulary
            dataset = feature_to_text_dataset.get_char_dataset(config=data_config)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            collate_fn=dataset._collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=cfg.get('num_workers', 0),
            pin_memory=cfg.get('pin_memory', False),
        )
    else:
        logging.info("Using raw audios as input...")
        config = {
            "manifest_filepath": cfg.dataset_manifest,
            "batch_size": cfg['batch_size'],
            "num_workers": cfg.get('num_workers', 0),
        }
        dataloader = asr_model._setup_transcribe_dataloader(config)

    logging.info(f"Transcribing...")

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    hypotheses = []
    all_hypotheses = []
    with profile_fn(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        t0 = time.time()
        with autocast():
            with torch.no_grad():
                with record_fn("infer_loop"):
                    for test_batch in tqdm(dataloader, desc="Transcribing"):
                        with record_fn("infer_model"):
                            if cfg.use_feature:
                                outputs = asr_model.forward(
                                    processed_signal=test_batch[0].to(map_location),
                                    processed_signal_length=test_batch[1].to(map_location),
                                )
                            else:
                                outputs = asr_model.forward(
                                    input_signal=test_batch[0].to(map_location),
                                    input_signal_length=test_batch[1].to(map_location),
                                )

                        with record_fn("infer_other"):
                            logits, logits_len = outputs[0], outputs[1]

                            current_hypotheses, all_hyp = decode_function(
                                logits, logits_len, return_hypotheses=return_hypotheses,
                            )

                            if return_hypotheses and not is_rnnt:
                                # dump log probs per file
                                for idx in range(logits.shape[0]):
                                    current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                                    if current_hypotheses[idx].alignments is None:
                                        current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence

                            hypotheses += current_hypotheses
                            if all_hyp is not None:
                                all_hypotheses += all_hyp
                            else:
                                all_hypotheses += current_hypotheses

                            del logits
                            del test_batch
        t1 = time.time()
    logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
    if cfg.profiling:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("--------------------------------------------------------------------\n")

    # Save output to manifest
    manifest_data = load_manifest(cfg.dataset_manifest)
    groundtruth = []
    for i in range(len(manifest_data)):
        manifest_data[i]["pred_text"] = hypotheses[i]
        groundtruth.append(clean_label(manifest_data[i]["text"]))
    save_manifest(manifest_data, cfg.output_filename)
    logging.info(f"Output saved at {cfg.output_filename}")

    if cfg.use_pure_noise:
        hypotheses = " ".join(hypotheses)
        words = hypotheses.split()
        chars = "".join(words)
        logging.info("-----------------------------------------")
        logging.info(f"Number of hallucinated characters={len(chars)}")
        logging.info(f"Number of hallucinated words={len(words)}")
        logging.info(f"Concatenated predictions: {hypotheses}")
        logging.info("-----------------------------------------")
    else:
        wer_score = word_error_rate(hypotheses=hypotheses, references=groundtruth)
        logging.info("-----------------------------------------")
        logging.info(f"WER={wer_score:.4f}")
        logging.info("-----------------------------------------")


def save_manifest(manifest_data, out_file):
    with Path(out_file).open("w") as fout:
        for item in manifest_data:
            fout.write(f"{json.dumps(item)}\n")


def load_manifest(manifest_file):
    data = []
    with Path(manifest_file).open("r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def clean_label(_str, num_to_words=True):
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    Parameters
    ----------
    _str : the original string
    Returns
    -------
    string
    """
    replace_with_space = [char for char in '/?*\",.:=?_{|}~¨«·»¡¿„…‧‹›≪≫!:;ː→']
    replace_with_blank = [char for char in '`¨´‘’“”`ʻ‘’“"‘”']
    replace_with_apos = [char for char in '‘’ʻ‘’‘']
    _str = _str.strip()
    _str = _str.lower()
    for i in replace_with_blank:
        _str = _str.replace(i, "")
    for i in replace_with_space:
        _str = _str.replace(i, " ")
    for i in replace_with_apos:
        _str = _str.replace(i, "'")
    if num_to_words:
        _str = convert_num_to_words(_str)
    return " ".join(_str.split())


def convert_num_to_words(_str):
    """
    Convert digits to corresponding words
    Parameters
    ----------
    _str : the original string
    Returns
    -------
    string
    """
    num_to_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    _str = _str.strip()
    words = _str.split()
    out_str = ""
    num_word = []
    for word in words:
        if word.isnumeric():
            num = int(word)
            while num:
                digit = num % 10
                digit_word = num_to_words[digit]
                num_word.append(digit_word)
                num = int(num / 10)
                if not (num):
                    num_str = ""
                    num_word = num_word[::-1]
                    for ele in num_word:
                        num_str += ele + " "
                    out_str += num_str + " "
                    num_word.clear()
        else:
            out_str += word + " "
    out_str = out_str.strip()
    return out_str


if __name__ == "__main__":
    main()
