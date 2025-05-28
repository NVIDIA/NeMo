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


import torch
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm, vlm

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.vlm.gemma3vl.data.mock import Gemma3VLMockDataModule
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback


NAME = "gemma3vl_4b"

HF_MODEL_NAME = "google/gemma-3-4b-it"


if __name__ == "__main__":
    llm.export_ckpt(
        path="/tmp/gemma3vl_4b/checkpoints/val_loss=0.00-step=4-consumed_samples=20.0-last",
        target="hf",
        output_path="/tmp/gemma3vl_4b_hf",
    )
