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
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.language_modeling.megatron.gpt_request_dataset import GPTRequestDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.app_state import AppState


"""
Usage:
    a. If you need to run model on a few prompts from the file:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --prompt .

    b. If you need to run model on a prompt from the CLI:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --tokens_to_generate=32 \
            --prompt=YOUR_PROMPT
"""

assert torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default="", required=True, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--path_to_file", type=str, default="", required=False, help="Path to file with prompts (a text to complete)"
    )
    parser.add_argument(
        "--prompt", type=str, default="", required=True, help="Prompt for the model (a text to complete)"
    )
    parser.add_argument(
        "--tokens_to_generate", type=int, default="64", required=False, help="How many tokens to add to prompt"
    )
    parser.add_argument(
        "--stop_after_sentence",
        type=bool,
        default="True",
        required=False,
        help="True/False: whether to stop after full sentence has been generated.",
    )
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument("--precision", default=32, help="PyTorch Lightning Trainer precision flag")

    args = parser.parse_args()

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(float(args.precision))

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=NLPDDPPlugin(), gpus=args.tensor_model_parallel_size, precision=args.precision)

    app_state = AppState()
    if args.tensor_model_parallel_size is not None and args.tensor_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size
        app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

    model = MegatronGPTModel.restore_from(restore_path=args.model_file, trainer=trainer)

    model.freeze()

    # defining type of request
    if args.path_to_file != "":
        data = []
        prompts = open(args.path_to_file, 'r')

        for prompt in prompts.readlines():
            request = {
                "prompt": prompt.split('\n')[0],
                "tokens_to_generate": args.tokens_to_generate,
                "stop_after_sentence": args.stop_after_sentence,
            }
            data.append(request)

        dataset = GPTRequestDataset(data, model.tokenizer)
        request_dl = DataLoader(dataset)
        response = trainer.predict(model, request_dl)
    else:
        request = [
            {
                "prompt": args.prompt,
                "tokens_to_generate": args.tokens_to_generate,
                "stop_after_sentence": args.stop_after_sentence,
            }
        ]
        dataset = GPTRequestDataset(request, model.tokenizer)
        request_dl = DataLoader(dataset)
        response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
