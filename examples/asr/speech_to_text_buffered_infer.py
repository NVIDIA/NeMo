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

import torch

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()
import json
import time
import os
from omegaconf import OmegaConf
import copy
import collections
import nemo.collections.asr as nemo_asr
import torch
from nemo.collections.asr.metrics.wer import word_error_rate

import numpy as np
import math
from nemo.collections.asr.parts.utils.streaming_utils import FeatureFrameBufferer, FrameBatchASR, get_samples, AudioFeatureIterator

def clean_label(_str):
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    Parameters
    ----------
    _str : the original string
    Returns
    -------
    string
    """
    if _str is None:
        return
    _str = _str.strip()
    _str = _str.lower()
    _str = _str.replace(".", "")
    _str = _str.replace(",", "")
    _str = _str.replace("?", "")
    _str = _str.replace("!", "")
    _str = _str.replace(":", "")
    _str = _str.replace("-", " ")
    _str = _str.replace("_", " ")
    _str = _str.replace("  ", " ")
    return _str


def get_wer_feat(mfst, frame_bufferer, asr, frame_len, tokens_per_chunk, delay, preprocessor_cfg, model_stride_in_secs, device):
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = collections.defaultdict(list)
    refs = []
    wer_dict = {}

    first = True
    with open(mfst, "r") as mfst_f:
        for l in mfst_f:
            frame_bufferer.reset()
            asr.reset()
            row = json.loads(l.strip())
            samples = get_samples(row['audio_filepath'])
            samples = np.pad(samples, (0, int(delay * model_stride_in_secs * preprocessor_cfg.sample_rate)))
            frame_reader = AudioFeatureIterator(samples, frame_len, preprocessor, device)
            frame_bufferer.set_frame_reader(frame_reader)
            asr.infer_logits()
            hyp = asr.transcribe(tokens_per_chunk, delay)
            hyps[(tokens_per_chunk, delay)].append(hyp)
            refs.append(row['text'])

    for key in hyps.keys():
        wer_dict[key] = word_error_rate(hypotheses=hyps[key], references=refs)
    return hyps[(tokens_per_chunk, delay)], refs, wer_dict

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to asr model .nemo file",
    )
    parser.add_argument("--test_manifest", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--total_buffer_in_secs", type=float, default=4.0, help="Length of buffer (chunk + left and right padding) in seconds ")
    parser.add_argument("--chunk_len_in_ms", type=int, default=1600, help="Chunk length in milliseconds")
    parser.add_argument("--output_path", type=str, help="path to output file", default=None)
    parser.add_argument("--model_stride", type=int, default=8, help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)

    res = {}
    feature_stride = cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * args.model_stride
    total_buffer = args.total_buffer_in_secs

    chunk_len = args.chunk_len_in_ms / 1000

    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    print(tokens_per_chunk, mid_delay)

    frame_bufferer = FeatureFrameBufferer(
        asr_model=asr_model,
        frame_len=chunk_len,
        batch_size=64,
        total_buffer=args.total_buffer_in_secs)

    frame_asr = FrameBatchASR(frame_bufferer,
                        asr_model=asr_model,
                        )

    hyps, refs, wer_dict = get_wer_feat(args.test_manifest, frame_bufferer, frame_asr, chunk_len, tokens_per_chunk, mid_delay,
                                        cfg.preprocessor, model_stride_in_secs, asr_model.device)
    for key in wer_dict.keys():
        logging.info(f"WER is {wer_dict[key]} when decoded with a delay of {mid_delay*model_stride_in_secs}s")

    if args.output_path is not None:

        fname = (os.path.splitext(os.path.basename(args.asr_model))[0] + "_" +
                 os.path.splitext(os.path.basename(args.test_manifest))[0] + "_" +
                 str(args.chunk_len_in_ms) + "_" + str(int(total_buffer * 1000)) + ".json")
        hyp_json = os.path.join(args.output_path, fname)
        os.makedirs(args.output_path, exist_ok=True)
        with open(hyp_json, "w") as out_f:
            for i, hyp in enumerate(hyps):
                record = {
                    "pred_text": hyp,
                    "text": refs[i],
                    "wer": round(word_error_rate(hypotheses=[hyp], references=[refs[i]]) * 100, 2),
                }
                out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
