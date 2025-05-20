# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from megatron.core.distributed import DistributedDataParallelConfig

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.recipes.finetune_default import nemo_resume

from nemo import lightning as nl
from nemo.collections import vlm, llm
from nemo.collections.llm.recipes.log.default import tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo.collections.vlm import CosmosNemotronRadioLlama8BConfig
from nemo.collections.vlm.vision.vision_transform import VisualProcessor
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.utils.exp_manager import TimingCallback

NAME = "cosmos_nemotron_8b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Cosmos Nemotron 8B (Radio v2.5 h + Llama3.1 8B) model configuration.

    Returns:
        run.Config[pl.LightningModule]: Cosmos Nemotron 8B model.
    """
    model = run.Config(vlm.CosmosNemotronModel, config=run.Config(CosmosNemotronRadioLlama8BConfig))
    return model

@run.cli.factory(target=llm.finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    resume_path: str = "TODO",
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    peft_scheme: Optional[str] = 'none',
) -> run.Partial:
    """
    Create a fine-tuning recipe for Cosmos Nemotron 8B (Radio v2.5 h + Llama3.1 8B) model.

    This function sets up a complete configuration for fine-tuning, including
    model, trainer, data, logging, optimization, and resumption settings.
    The recipe uses LoRA (Low-Rank Adaptation) for efficient fine-tuning, unless peft_scheme is set to None.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        resume_path (str): Path to the NeMo checkpoint
        name (str): Name of the fine-tuning run.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        peft_scheme (Optional[str]): Name of the peft scheme to use for fine-tuning.
            Allowed values: 'lora'/'dora'/'none'/None.
    Returns:
        run.Partial: Partial configuration for fine-tuning.

    Examples:
        CLI usage:
            $ nemo llm finetune --factory cosmos_nemotron_8b

        Python API usage:
            >>> recipe = finetune_recipe(name="cosmos_nemotron_8b", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses the SQuAD dataset for fine-tuning. For more information
        on fine-tuning LLMs with NeMo, see the fine-tuning guide in the
        `examples/llm/finetune/` directory.
    """

    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        encoder_pipeline_model_parallel_size=0,
        sequence_parallel=True,
        pipeline_dtype=torch.bfloat16,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        devices=num_gpus_per_node,
        limit_val_batches=10,
        log_every_n_steps=1,
        max_steps=5190,
        num_nodes=num_nodes,
        plugins=bf16_mixed(),
        strategy=strategy,
        val_check_interval=1000,
        callbacks=[
            run.Config(TimingCallback),
            run.Config(MegatronCommOverlapCallback,
                tp_comm_overlap=False,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
           ),
        ],
    )
    tokenizer = run.Config(AutoTokenizer, "meta-llama/Llama-3.1-8B-Instruct")
    image_processor = run.Config(
        VisualProcessor,
        crop_height=512,
        crop_width=512,
        use_tiling=True,
        max_num_tiles=12,
        use_thumbnail=True,
        augment=False,
        vision_model_type="radio",
    )
    recipe = run.Partial(
        llm.finetune,
        model=model(),
        trainer=trainer,
        data=run.Config(
            vlm.NevaMockDataModule,
            seq_length=16384,
            global_batch_size=128,
            micro_batch_size=1,
            tokenizer=tokenizer,
            image_processor=image_processor,
            num_workers=4,
            packed_sequence=False,
            pixel_shuffle_ratio=0.5,
            num_image_embeddings_per_tile=1024,
            num_tiles_per_image=5,
        ),
        log=llm.default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=2.0e-06, min_lr=1.0e-07, warmup_steps=150),
        resume=nemo_resume(resume_path),
    )
    if peft_scheme is None or peft_scheme.lower() == 'none':
        recipe.model.config.recompute_granularity = "selective"
    elif peft_scheme.lower() == 'lora':
        recipe.peft = vlm.peft.LoRA(
            target_modules=[
                "*.language_model.*.linear_qkv",
                "*.language_model.*.linear_proj",
                "*.language_model.*.linear_fc1",
                "*.language_model.*.linear_fc2",
            ],
            freeze_language_model=True,
            freeze_vision_model=False,
            freeze_vision_projection=False,
            dim=32,
        )
    else:
        raise ValueError(f"Unrecognized peft scheme: {peft_scheme}")

    return recipe