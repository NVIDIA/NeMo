# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from argparse import ArgumentParser

import torch

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import ONNXGreedyBatchedRNNTInfer
from nemo.utils import logging


"""
Export nemo asr model to onnx 

# Compare a NeMo and ONNX model
python infer_transducer_onnx.py \
    --nemo_model="<path to a .nemo file>" \
    OR
    --pretrained_model="<name of a pretrained model>" 
    --export

"""


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--nemo_model", type=str, default=None, required=False, help="Path to .nemo file",
    )
    parser.add_argument(
        '--pretrained_model', type=str, default=None, required=False, help='Name of a pretrained NeMo file'
    )
    parser.add_argument('--onnx_model', type=str, default=None, required=False, help="Path to onnx model")
  
    args = parser.parse_args()
    return args


def assert_args(args):
    if args.nemo_model is None and args.pretrained_model is None:
        raise ValueError(
            "`nemo_model` or `pretrained_model` must be passed ! It is required for decoding the RNNT tokens "
            "and ensuring predictions match between Torch and ONNX."
        )

    if args.nemo_model is not None and args.pretrained_model is not None:
        raise ValueError(
            "`nemo_model` and `pretrained_model` cannot both be passed ! Only one can be passed to this script."
        )


class EncDecModel(torch.nn.Module):
    def __init__(self, encoder_module, decoder_module):
        super().__init__()
        self.encoder = encoder_module
        self.decoder = decoder_module
    
    def forward(self, audio_signal, length):
        enc_out, encoded_length = self.encoder(audio_signal=audio_signal, length=length)
        dec_out = self.decoder(encoder_output=enc_out)
        return dec_out, encoded_length


def export_model(args, nemo_model):
    nemo_model.preprocessor.featurizer.dither = 0.0
    nemo_model.preprocessor.featurizer.pad_to = 0
    device = nemo_model.device
    input_signal = torch.randn(1, 16000, dtype=torch.float32).to(device)
    input_signal_length = torch.tensor([16000], dtype=torch.int32).to(device)
    processed_audio, processed_audio_len = nemo_model.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
    model = EncDecModel(nemo_model.encoder, nemo_model.decoder)
    torch.onnx.export(model, (processed_audio, processed_audio_len), 
                      args.onnx_model, opset_version=12, verbose=True,
                      input_names=['audio_signal', 'length'], 
                      output_names=['log_probs', 'encoded_length'],
                      dynamic_axes={'audio_signal': {0: 'batch_size', 2: 'time'}, 
                                    'length': {0: 'batch_size'},
                                    'log_probs': {0: 'batch_size', 1: 'time'},
                                    'encoded_length': {0: 'batch_size'}})
    
def main():
    args = parse_arguments()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate pytorch model
    if args.nemo_model is not None:
        nemo_model = args.nemo_model
        nemo_model = ASRModel.restore_from(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    elif args.pretrained_model is not None:
        nemo_model = args.pretrained_model
        nemo_model = ASRModel.from_pretrained(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    else:
        raise ValueError("Please pass either `nemo_model` or `pretrained_model` !")

    if torch.cuda.is_available():
        nemo_model = nemo_model.to('cuda')

    export_model(args, nemo_model)

    

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
