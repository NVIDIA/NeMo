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

import json
from argparse import ArgumentParser
from torch.utils.data import Dataset

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from nemo.collections.nlp.models.language_modeling.megatron_gpt_eval_model import MegatronGPTEvalModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.collections.nlp.modules.common.text_generation_utils import generate

"""
Usage:
    a. If you need to run model on a few prompts from the file:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --batch_size=16 \

    b. If you need to run model on a prompt from the CLI:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --tokens_to_generate=32 \
            --prompt=YOUR_PROMPT

    c. If you don't need to generate tokens and need model to compute logprobs:
         python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --batch_size=16 \
            --compute_logprobs=True \
            --prompt .

    d. If you need to run a prompt-tuned model on a few prompts from a file:
        python megatron_gpt_eval.py \
            --use_soft_prompts \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --batch_size=16 \

        The path_to_file containing the model prompts should be a json with prompts in the format:
            {'prompt_tag': tag1, 'text': prompt1}
            {'prompt_tag': tag1, 'text': prompt2}
            {'prompt_tag': tag3, 'text': prompt3}

    e. If you need to run a prompt-tuned model on a prompt from the CLI:
        python megatron_gpt_eval.py \
            --use_soft_prompts \
            --model_file=PATH_TO_MODEL \
            --tokens_to_generate=32 \
            --prompt_tag=PROMPT_TAG_STRING \
            --prompt=YOUR_PROMPT
"""

assert torch.cuda.is_available()


class RequestDataSet(Dataset):

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return ''

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default="", required=True, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--path_to_file", type=str, default="", required=False, help="Path to file with prompts (a text to complete)"
    )
    parser.add_argument(
        "--prompt", type=str, default="", required=False, help="Prompt for the model (a text to complete)"
    )
    parser.add_argument(
        "--prompt_tag", type=str, default="", required=False, help="Prompt tag string for task specific soft prompt"
    )
    parser.add_argument(
        "--tokens_to_generate", type=int, default="1", required=False, help="How many tokens to add to prompt"
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
    parser.add_argument(
        "--pipeline_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="Evaluation batch_size")
    parser.add_argument(
        "--compute_logprobs", type=bool, default=False, required=False, help="Method for logprobs computation"
    )

    args = parser.parse_args()

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(float(args.precision))

    # trainer required for restoring model parallel models
    trainer = Trainer(
        plugins=NLPDDPPlugin(),
        devices=args.tensor_model_parallel_size * args.pipeline_model_parallel_size,
        accelerator='gpu',
        precision=args.precision,
    )

    app_state = AppState()
    if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            _,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=args.tensor_model_parallel_size,
            pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
        )

    model = MegatronGPTEvalModel.restore_from(restore_path=args.model_file, trainer=trainer)
    model.freeze()
    ds = RequestDataSet()

    request_dl = DataLoader(dataset=ds, batch_size=2)


    # turn off the activation checkpoint method
    model.model.language_model.encoder.activations_checkpoint_method=None
    #generate(model, ["", ""], 30)
    # For GPT models that have had soft prompt tuning but you don't want to use any soft prompts
    response = trainer.predict(model, request_dl)

    #print("***************************")
    #print(response)
    #print("***************************")
    #if args.prompt and not args.compute_logprobs:

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
