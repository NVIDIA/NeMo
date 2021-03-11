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
import argparse

from nemo.collections.asr.models import EncDecClassificationModel, EncDecCTCModel, EncDecSpeakerLabelModel
from nemo.utils import logging


def get_parser():
    parser = argparse.ArgumentParser(description="Convert .nemo file to encoder decoder onnx files")
    parser.add_argument(
        "--nemo_file", default=None, type=str, required=True, help="Path to .nemo file",
    )
    parser.add_argument(
        "--onnx_encoder", default=None, type=str, required=True, help="Path to the onnx encoder output.",
    )
    parser.add_argument(
        "--onnx_decoder", default=None, type=str, required=True, help="Path to the onnx decoder output.",
    )
    parser.add_argument(
        "--model_type",
        default='asr',
        type=str,
        choices=['asr', 'speech_label', 'speaker'],
        help="Type of decoder used by the model.",
    )
    return parser


def main(
    nemo_file, onnx_encoder, onnx_decoder, model_type='asr',
):
    if model_type == 'asr':
        logging.info("Preparing encoder decoder for ASR model")
        model = EncDecCTCModel.restore_from(nemo_file)
    elif model_type == 'speech_label':
        logging.info("Preparing encoder decoder for Speech Label Classification model")
        model = EncDecClassificationModel.restore_from(nemo_file)
    elif model_type == 'speaker':
        logging.info("Preparing encoder decoder for Speaker Recognition model")
        model = EncDecSpeakerLabelModel.restore_from(nemo_file)
    else:
        raise NameError("Available model names are asr, speech_label and speaker ")

    logging.info("Writing onnx encoder and decoder onnx files")
    model.encoder.export(onnx_encoder)
    model.decoder.export(onnx_decoder, onnx_opset_version=10)
    logging.info("succesfully ported onnx files")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(
        args.nemo_file, args.onnx_encoder, args.onnx_decoder, model_type=args.model_type,
    )
