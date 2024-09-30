# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import torch
from tqdm import tqdm

from nemo.collections.llm.quantization import Quantizer, get_calib_data_iter
# TODO: Support PP
# TODO: Inference TP/PP != Calibration TP/PP

# TODO: maybe use llm.generate (#10471)
def forward_loop(model):
    tokenizer = model.tokenizer
    dataloader = get_calib_data_iter()
    dataloader = [data for data in dataloader]

    for batch in tqdm(dataloader):
        batch = [tokenizer.text_to_ids(text) for text in batch]
        max_len = max([len(text) for text in batch])
        batch = [ids + (max_len - len(ids)) * [tokenizer.eos] for ids in batch]
        position_ids = torch.arange(max_len, device=model.device).expand((len(batch), max_len))
        batch = torch.tensor(batch, device=model.device)
        model_input = {
            "input_ids": batch,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        model(**model_input)


def main():
    parser = Quantizer.create_argparser()
    params = parser.parse_args(sys.argv[1:])
    params = Quantizer.postprocess_argparse(params)

    quantizer = Quantizer(params.quantization_config, params.export_config)
    model = quantizer.load_quantizable_model(params.nemo_checkpoint, params.tensor_parallelism_size)

    if params.quant_algo != "no_quant":
        model = quantizer.quantize(model, forward_loop)
    quantizer.export(model)


if __name__ == '__main__':
    main()
