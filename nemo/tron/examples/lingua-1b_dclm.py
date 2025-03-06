import math
from functools import partial

import torch
import torch.distributed
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo.collections import llm
from nemo.collections.llm.gpt.model.base import gpt_data_step
from nemo.tron.api import megatron_pretrain
from nemo.tron.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from nemo.tron.data.dataset import get_blend_and_blend_per_split
from nemo.tron.state import GlobalState

# define spiky loss as a variation of 20% or more
SPIKY_LOSS_PERC = 0.2


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    state = GlobalState()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if state.cfg.model_config.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * state.cfg.model_config.context_parallel_size,
        local_num_tokens,
        {"lm loss": (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = GlobalState().timers

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    batch = gpt_data_step(data_iterator)
    if "attention_mask" not in batch:
        batch["attention_mask"] = None

    tokens, labels, loss_mask, attention_mask, position_ids = (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
    )
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


if __name__ == "__main__":
    global_batch_size = 256
    micro_batch_size = 1
    seq_length = 4096

    # Model
    model_cfg = llm.Llama32Config1B(
        num_layers=25,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=16,
        ffn_hidden_size=6144,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        attention_softmax_in_fp32=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        seq_length=seq_length,
        make_vocab_size_divisible_by=128,
    )

    # Dataset
    blend, blend_per_split = get_blend_and_blend_per_split(
        data_paths=[f"/path/to/data/dclm_{i:02d}_text_document" for i in range(1, 51)]
    )

    tokens = 60_000_000_000
    max_steps = math.ceil(tokens / model_cfg.seq_length / global_batch_size)

    # Config Container
    cfg = ConfigContainer(
        model_config=model_cfg,
        train_config=TrainingConfig(
            train_iters=max_steps,
            eval_interval=2000,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            exit_signal_handler=True,
        ),
        optimizer_config=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-5,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            lr=3e-3,
            weight_decay=0.01,
            min_lr=1e-6,
        ),
        scheduler_config=SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=5000,
            lr_warmup_init=0.0,
            lr_decay_iters=max_steps,
            override_opt_param_scheduler=True,
        ),
        ddp_config=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset_config=GPTDatasetConfig(
            blend=blend,
            blend_per_split=blend_per_split,
            random_seed=2788,
            sequence_length=model_cfg.seq_length,
            path_to_cache="/hubs/data/index_mapping_dclm_1_0",
            reset_position_ids=False,
            create_attention_mask=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            num_dataset_builder_threads=1,
            split="9999,8,2",
            data_sharding=True,
            dataloader_type="single",
            num_workers=2,
        ),
        logger_config=LoggerConfig(
            wandb_project="nemo_custom_pretraining_loop",
            wandb_entity="nvidia",
            wandb_exp_name=f"lingua_dclm_full_1b_gbs_{global_batch_size}_20250303",
            wandb_save_dir="/nemo_run/wandb",
            tensorboard_dir="/nemo_run/tensorboard",
            log_timers_to_tensorboard=True,
            log_validation_ppl_to_tensorboard=True,
            tensorboard_log_interval=10,
            timing_log_level=2,
            log_progress=True,
            log_interval=10,
            logging_level="INFO",
        ),
        tokenizer_config=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="/path/to/hf-tokenizer/",
        ),
        checkpoint_config=CheckpointConfig(
            save_interval=10000,
            save="/nemo_run/checkpoints",
            load="/nemo_run/checkpoints",
            async_save=True,
            fully_parallel_save=True,
        ),
        rng_config=RNGConfig(seed=2888),
    )

    megatron_pretrain(cfg, forward_step)
