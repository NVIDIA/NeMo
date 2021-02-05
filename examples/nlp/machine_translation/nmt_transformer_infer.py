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
Given NMT model's .nemo file, this script can be used to translate text.
USAGE Example:
1. Obtain text file in src language. You can use sacrebleu to obtain standard test sets like so:
    sacrebleu -t wmt14 -l de-en --echo src > wmt14-de-en.src
2. Translate:
    python nmt_transformer_infer.py --model=[Path to .nemo file] --srctext=wmt14-de-en.src --tgtout=wmt14-de-en.pre
"""


from argparse import ArgumentParser

import torch

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--srctext", type=str, required=True, help="")
    parser.add_argument("--tgtout", type=str, required=True, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--beam_size", type=int, default=4, help="")
    parser.add_argument("--len_pen", type=float, default=0.6, help="")
    parser.add_argument("--max_delta_length", type=int, default=5, help="")
    parser.add_argument("--target_lang", type=str, default=None, help="")
    parser.add_argument("--source_lang", type=str, default=None, help="")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=args.model)
        src_text = []
        tgt_text = []
    else:
        raise NotImplemented(f"Only support .nemo files, but got: {args.model}")

    model.beam_search.beam_size = args.beam_size
    model.beam_search.len_pen = args.len_pen
    model.beam_search.max_delta_length = args.max_delta_length

    if torch.cuda.is_available():
        model = model.cuda()

    logging.info(f"Translating: {args.srctext}")

    count = 0
    with open(args.srctext, 'r') as src_f:
        for line in src_f:
            src_text.append(line.strip())
            if len(src_text) == args.batch_size:
                res = model.translate(text=src_text, source_lang=args.source_lang, target_lang=args.target_lang)
                if len(res) != len(src_text):
                    print(len(res))
                    print(len(src_text))
                    print(res)
                    print(src_text)
                tgt_text += res
                src_text = []
            count += 1
            # if count % 300 == 0:
            #    print(f"Translated {count} sentences")
        if len(src_text) > 0:
            tgt_text += model.translate(text=src_text, source_lang=args.source_lang, target_lang=args.target_lang)

    with open(args.tgtout, 'w') as tgt_f:
        for line in tgt_text:
            tgt_f.write(line + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
