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


from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from pytorch_lightning.trainer.trainer import Trainer
from nemo.utils import logging
from argparse import ArgumentParser

import torch

from nemo.utils import logging


assert torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default="", required=True, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--prompt", type=str, default="", required=True, help="Prompt for the model (a text to complete)"
    )
    parser.add_argument(
        "--tokens_to_generate", type=int, default="64", required=True, help="How many tokens to add to prompt"
    )
    parser.add_argument(
        "--stop_after_sentence",
        type=bool,
        default="True",
        required=False,
        help="True/False: whether to stop after full sentence has been generated.",
    )

    args = parser.parse_args()
    torch.set_grad_enabled(False)

    # trainer required for restoring model parallel models
    trainer = Trainer()
    model = MegatronGPTModel.restore_from(restore_path=args.model_file, trainer=trainer)
    res = model.complete(
        {
            "prompt": args.prompt,
            "tokens_to_generate": args.tokens_to_generate,
            "stop_after_sentence": args.stop_after_sentence,
        }
    )
    print("***************************")
    print(res['completion']['text'])
    print("***************************")
    logging.info(f"Generation stopped because: {res['completion']['stop reason']}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
