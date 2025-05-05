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
Converts BERT NeMo0.* checkpoints to NeMo1.0 format.
"""

from argparse import ArgumentParser

import torch

parser = ArgumentParser()
parser.add_argument("--bert_encoder", required=True, help="path to BERT encoder, e.g. /../BERT-STEP-2285714.pt")
parser.add_argument(
    "--bert_token_classifier",
    required=True,
    help="path to BERT token classifier, e.g. /../BertTokenClassifier-STEP-2285714.pt",
)
parser.add_argument(
    "--bert_sequence_classifier",
    required=False,
    default=None,
    help="path to BERT sequence classifier, e.g /../SequenceClassifier-STEP-2285714.pt",
)
parser.add_argument(
    "--output_path", required=False, default="converted_model.pt", help="output path to newly converted model"
)
args = parser.parse_args()

bert_in = torch.load(args.bert_encoder)
tok_in = torch.load(args.bert_token_classifier)
if args.bert_sequence_classifier:
    seq_in = torch.load(args.bert_sequence_classifier)

new_dict = {}
new_model = {"state_dict": new_dict}
for k in bert_in:
    new_name = k.replace("bert.", "bert_model.")
    new_dict[new_name] = bert_in[k]

for k in tok_in:
    new_name = "mlm_classifier." + k
    new_dict[new_name] = tok_in[k]

if args.bert_sequence_classifier:
    for k in seq_in:
        new_name = "nsp_classifier." + k
        new_dict[new_name] = seq_in[k]

torch.save(new_model, args.output_path)
