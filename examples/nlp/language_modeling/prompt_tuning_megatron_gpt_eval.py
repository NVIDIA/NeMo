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
from omegaconf.omegaconf import OmegaConf, open_dict

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

from nemo.collections.nlp.data.language_modeling.megatron.gpt_request_dataset import GPTRequestDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.app_state import AppState

assert torch.cuda.is_available()

# +
precision = 16
tensor_model_parallel_size = 1
model_file = "prompt_tuned_megatron_gpt.nemo"

# cast precision to int if 32 or 16
if precision in ["32", "16"]:
    precision = int(float(precision))

# trainer required for restoring model parallel models
trainer = Trainer(plugins=NLPDDPPlugin(), gpus=tensor_model_parallel_size, precision=precision)

app_state = AppState()

if tensor_model_parallel_size is not None and tensor_model_parallel_size > 1:
    app_state.model_parallel_size = tensor_model_parallel_size
    app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

cfg = OmegaConf.load("conf/megatron_gpt_config.yaml")

# Set current model params
cfg.model.encoder_seq_length = 2048

# Set prompt tuning params
cfg.model.optim.lr = 2e-4
cfg.model.optim.sched.min_lr = 2e-6
cfg.model.use_soft_prompts = True
cfg.model.prompt_length = 10
cfg.model.data.train_ds = 'prompt_tuning_ner_train.json'
cfg.model.data.valid_ds = 'prompt_tuning_ner_val.json'
cfg.model.data.test_ds = 'prompt_tuning_ner_test.json'
cfg.model.data.batch_size = 32
cfg.model.data.data_prefix = None
cfg.model.existing_prompt_tags = ["NER-Yes-No", "NER-Complete"]
cfg.model.optim.sched.warmup_steps = 50
cfg.model.optim.sched.constant_steps = 100
cfg.trainer.max_steps = 200
cfg.restore_from_path = 'prompt_tuning_megatron_gpt2.nemo'
    
model = MegatronGPTModel.restore_from(model_file, cfg.model, trainer=trainer)

model.freeze()

# +
request = [
            {
                "prompt_tag": "NER-Complete",
                "prompt": 'find entities: "Downregulation of survivin expression and concomitant induction of apoptosis by celecoxib and its non-cyclooxygenase-2-inhibitory analog, dimethyl-celecoxib (DMC), in tumor cells in vitro and in vivo" answer: "',
                "tokens_to_generate": 64,
                "stop_after_sentence": True,
            }
        ]

dataset = GPTRequestDataset(request, model.tokenizer)
request_dl = DataLoader(dataset)
response = trainer.predict(model, request_dl)

print(response)


# -

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


