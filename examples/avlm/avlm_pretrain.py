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

"""
Example:

"""

import argparse

import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # pylint: disable=C0115,C0116

    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    num_workers = args.num_workers

    if args.data_type == "energon":
        from nemo.collections.avlm.data.energon import AVLMDataModule
        from nemo.collections.avlm.data.energon import AVLMSampleConfig
        from nemo.collections.avlm.data.energon import AVLMTaskEncoder

        data_path = args.data_path

        avlm_sample_config = AVLMSampleConfig()
        # Setting system prompt to empty string
        avlm_sample_config.conversation_template_config.system = ''

        task_encoder = AVLMTaskEncoder(
            multimodal_sample_config=avlm_sample_config,
        )
        data = AVLMDataModule(
            path=data_path,
            num_workers=num_workers,
            micro_batch_size=mbs,
            global_batch_size=gbs,
            multimodal_sample_config=avlm_sample_config,
            task_encoder=task_encoder,
        )
    else:
        raise ValueError(f"Data type {args.data_type} not supported")

    samples = []
    loader = data.train_dataloader()
    for sample in loader: 
        import pdb;pdb.set_trace()
        samples.append(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llava Next Pretraining Script")

    # Argument parsing
    parser.add_argument("--data_type", type=str, required=False, default="energon", help="mock | energon")
    parser.add_argument("--data_path", type=str, required=False, default=None, help="Path to the dataset with data.yaml file."
        " More details about Energon data preparation: https://nvidia.github.io/Megatron-Energon/index.html")
    parser.add_argument("--gbs", type=int, required=False, default=32, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=4, help="Micro batch size")
    parser.add_argument("--num_workers", type=int, required=False, default=32, help="Number of workers for data loading")

    args = parser.parse_args()
    main(args)
