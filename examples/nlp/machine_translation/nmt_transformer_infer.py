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

from argparse import ArgumentParser

import torch

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--text2translate", type=str, required=True, help="")
    parser.add_argument("--output", type=str, required=True, help="")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = nemo_nlp.models.TransformerMTModel.restore_from(restore_path=args.model)
    elif args.model.endswith(".ckpt"):
        logging.info("Attempting to initialize from .ckpt file")
        model = nemo_nlp.models.TransformerMTModel.load_from_checkpoint(checkpoint_path=args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    logging.info(f"Translating: {args.text2translate}")
    txt_to_translate = []
    with open(args.text2translate, 'r') as fin:
        for line in fin:
            txt_to_translate.append(line.strip())
    print(txt_to_translate)
    translation = model.translate(text=txt_to_translate)
    with open(args.output, 'w') as fout:
        for txt in translation:
            fout.write(txt + "\n")
    logging.info("all done")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
