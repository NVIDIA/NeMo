# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
A script to convert a Nemo ASR Hybrid model file (.nemo) to a Nemo ASR CTC or RNNT model file (.nemo)

This allows you to train a RNNT-CTC Hybrid model, but then convert to a pure CTC or pure RNNT model for use
in Riva. Works just fine with nemo2riva, HOWEVER, Riva doesn't support AggTokenizer, but nemo2riva
does, so be careful that you do not convert a model with AggTokenizer and then use that in Riva
as it will not work.

Usage: python convert_nemo_asr_hybrid_to_ctc.py -i /path/to/hybrid.nemo -o /path/to/saved_ctc_model.nemo -m [ctc|rnnt]

"""


import argparse
import os

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import (
    ASRModel,
    EncDecCTCModel,
    EncDecCTCModelBPE,
    EncDecRNNTBPEModel,
    EncDecRNNTModel,
)
from nemo.utils import logging


def extract_model_ctc(args, hybrid_model):
    BPE = False
    ctc_class = EncDecCTCModel
    if 'tokenizer' in hybrid_model.cfg.keys():
        BPE = True
        ctc_class = EncDecCTCModelBPE

    hybrid_model_cfg = OmegaConf.to_container(hybrid_model.cfg)

    new_cfg = hybrid_model_cfg.copy()
    new_cfg['ctc_reduction'] = hybrid_model_cfg['aux_ctc']['ctc_reduction']
    new_cfg['decoder'] = hybrid_model_cfg['aux_ctc']['decoder']
    del new_cfg['compute_eval_loss']
    del new_cfg['model_defaults']
    del new_cfg['joint']
    del new_cfg['decoding']
    del new_cfg['aux_ctc']
    del new_cfg['loss']
    if BPE and 'labels' in new_cfg:
        del new_cfg['labels']
    elif (not BPE) and 'tokenizer' in new_cfg:
        del new_cfg['tokenizer']
    del new_cfg['target']
    del new_cfg['nemo_version']

    new_cfg_oc = OmegaConf.create(new_cfg)
    ctc_model = ctc_class.restore_from(
        args.input, map_location=torch.device('cpu'), override_config_path=new_cfg_oc, strict=False
    )

    assert all(
        [
            torch.allclose(hybrid_model.state_dict()[x], ctc_model.state_dict()[x])
            for x in hybrid_model.state_dict().keys()
            if x.split('.')[0] in ['preprocessor', 'encoder']
        ]
    ), "Encoder and preprocessor state dicts don't match!"

    ctc_model.decoder.load_state_dict(hybrid_model.ctc_decoder.state_dict())

    assert all(
        [
            torch.allclose(hybrid_model.ctc_decoder.state_dict()[x], ctc_model.decoder.state_dict()[x])
            for x in hybrid_model.ctc_decoder.state_dict().keys()
        ]
    ), "Decoder state_dict load failed!"

    assert isinstance(ctc_model, ctc_class), "Extracted CTC model is of the wrong expected class!"

    return ctc_model


def extract_model_rnnt(args, hybrid_model):
    BPE = False
    rnnt_class = EncDecRNNTModel
    if 'tokenizer' in hybrid_model.cfg.keys():
        BPE = True
        rnnt_class = EncDecRNNTBPEModel

    hybrid_model_cfg = OmegaConf.to_container(hybrid_model.cfg)

    new_cfg = hybrid_model_cfg.copy()
    del new_cfg['aux_ctc']
    if BPE and 'labels' in new_cfg:
        del new_cfg['labels']
    elif (not BPE) and 'tokenizer' in new_cfg:
        del new_cfg['tokenizer']
    del new_cfg['target']
    del new_cfg['nemo_version']

    new_cfg_oc = OmegaConf.create(new_cfg)
    rnnt_model = rnnt_class.restore_from(
        args.input, map_location=torch.device('cpu'), override_config_path=new_cfg_oc, strict=False
    )

    assert all(
        [
            torch.allclose(hybrid_model.state_dict()[x], rnnt_model.state_dict()[x])
            for x in hybrid_model.state_dict().keys()
            if x.split('.')[0] in ['preprocessor', 'encoder', 'decoder', 'joint']
        ]
    ), "State dict values mismatch, something went wrong!"

    assert isinstance(rnnt_model, rnnt_class), "Extracted RNNT model is of the wrong expected class!"

    return rnnt_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to Nemo Hybrid model .nemo file')
    parser.add_argument('-o', '--output', required=True, type=str, help='path and name of output .nemo file')
    parser.add_argument(
        '-m',
        '--model',
        required=False,
        type=str,
        default='ctc',
        choices=['ctc', 'rnnt'],
        help='whether to output a ctc or rnnt model from the hybrid',
    )
    parser.add_argument('--cuda', action='store_true', help='put Nemo model onto GPU prior to savedown')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.critical(f'Input file [ {args.input} ] does not exist or cannot be found. Aborting.')
        exit(255)

    hybrid_model = ASRModel.restore_from(args.input, map_location=torch.device('cpu'))

    if args.model == 'ctc':
        output_model = extract_model_ctc(args, hybrid_model)
    elif args.model == 'rnnt':
        output_model = extract_model_rnnt(args, hybrid_model)
    else:
        logging.critical(
            f"the model arg must be one of 'ctc' or 'rnnt', received unknown value: '{args.model}'. Aborting."
        )
        exit(255)

    if args.cuda and torch.cuda.is_available():
        output_model = output_model.cuda()

    output_model.save_to(args.output)
    logging.info(f'Converted {args.model.upper()} model was successfully saved to {args.output}')
