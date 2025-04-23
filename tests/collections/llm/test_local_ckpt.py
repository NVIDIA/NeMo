# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# ... (License header remains the same)

"""
Test NeMo 2.0 with local checkpointing from NVRx,
including crash simulation and resumption verification.

This test requires a shared filesystem for --log-dir if used on multiple nodes.
"""

import argparse
import logging
import os
import re
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from lightning.pytorch.callbacks import Callback
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.collections.llm.recipes.log.default import get_global_step_from_global_checkpoint_path
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.local_ckpt import update_trainer_local_checkpoint_io
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.import_utils import safe_import

logger = logging.getLogger(__name__)

res_module, HAVE_RES = safe_import("nvidia_resiliency_ext.ptl_resiliency")


@dataclass
class Llama3Config145M(Llama3Config):
    rotary_base: int = 500_000
    seq_length: int = 8192  # Reduced for faster testing if needed
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16


class CrashException(Exception):
    """Custom exception for triggering simulated crashes in tests."""

    pass


class CrashCallback(Callback):
    """
    Callback to simulate a crash at a specific step.
    Uses trainer.global_step for accurate step counting.
    """

    def __init__(self, crash_step: Optional[int]):
        if crash_step is not None and crash_step <= 0:
            raise ValueError("crash_step must be a positive integer or None")
        self.crash_step = crash_step
        # No internal step counter needed - rely on trainer.global_step
        if self.crash_step:
            # Use logger - assumes logger is configured before instantiation
            logger.info(f"CrashCallback initialized. Will simulate crash if global_step reaches {self.crash_step}")
        else:
            logger.info("CrashCallback initialized. Crash simulation is disabled (crash_step is None).")

    def on_train_batch_end(self, trainer: "pl.Trainer", *args) -> None:
        # Check if crash simulation is enabled for this run
        if not self.crash_step:
            return

        # Get reliable global step and rank from the trainer
        current_global_step = trainer.global_step

        # Crash condition: Check step number and ensure only rank 0 crashes
        # Crash occurs *after* batch 'crash_step' finishes forward/backward
        if current_global_step == self.crash_step:
            msg = f"Simulating crash via CrashCallback at global_step {current_global_step}!"
            logger.error(msg)
            raise CrashException(msg)


def get_trainer(args, callbacks, plugins, strategy) -> nl.Trainer:
    """Creates a PyTorch Lightning Trainer instance."""
    return nl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        plugins=plugins,
        strategy=strategy,
        enable_progress_bar=False,
    )


def get_megatron_strategy(args, async_save: bool = True) -> nl.MegatronStrategy:
    """Creates a NeMo MegatronStrategy instance."""
    return nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=None,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        ckpt_async_save=async_save,
        ckpt_parallel_load=False,
        ddp=DistributedDataParallelConfig(),
    )


def get_optimizer(bf16_enabled: bool = True) -> MegatronOptimizerModule:
    """Creates a MegatronOptimizerModule instance."""
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        clip_grad=1.0,
        log_num_zeros_in_grad=False,
        timers=None,
        bf16=bf16_enabled,
        use_distributed_optimizer=False,
    )
    return MegatronOptimizerModule(config=opt_config)


def get_my_local_ckpt_node_dir(log_dir: str, rank: int) -> Path:
    """Gets the expected local checkpoint directory path."""
    return Path(log_dir) / "local_ckpt" / socket.gethostname() / str(rank)


def find_latest_local_ckpt_step(local_ckpt_node_dir: Path, global_rank: int) -> Optional[int]:
    """Finds the step number of the latest local checkpoint ON THIS NODE."""
    latest_step = -1

    if not local_ckpt_node_dir.is_dir():
        logger.warning(f"Local checkpoint node directory not found: {local_ckpt_node_dir}")
        return None

    pattern = re.compile(fr"iter_(\d+)_{global_rank}_local")
    for item in local_ckpt_node_dir.iterdir():
        match = pattern.match(item.name)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step

    result = latest_step if latest_step != -1 else None
    if result is not None:
        logger.info(f"[Rank {global_rank}] Found latest local checkpoint step {result} in {local_ckpt_node_dir}")
    else:
        logger.info(f"[Rank {global_rank}] No local checkpoints found matching pattern in {local_ckpt_node_dir}")
    return result


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser."""
    parser = argparse.ArgumentParser(description="Llama3 Local Ckpt Crash Test")
    parser.add_argument("--log-dir", type=str, required=True, help="Filesystem output dir.")
    parser.add_argument("--num-nodes", type=int, default=1, help="Total nodes in the job (usually from SLURM)")
    parser.add_argument("--devices", type=int, default=2, help="GPUs per node (usually from SLURM)")
    parser.add_argument('--max-steps', type=int, default=200, help="Total steps for training")
    parser.add_argument('--checkpoint-interval', type=int, default=80, help="Global checkpoint interval")
    parser.add_argument('--local-checkpoint-interval', type=int, default=45, help="Local checkpoint interval")
    parser.add_argument('--val-check-interval', type=int, default=80, help="Validation interval")
    parser.add_argument('--limit_val_batches', type=int, default=10, help="Validation batches limit")
    parser.add_argument("--async-save", action="store_true", help="Use async global ckpt save")
    parser.add_argument("--crash-step", type=int, default=100, help="Global step for Rank 0 to simulate crash")
    parser.add_argument("--cleanup-log-dir", action="store_true", help="Rank 0 cleans up log dir before starting")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    # --- Configure Logging ---
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    # Basic config - logs to stderr by default. Log to stdout for potentially cleaner capture.
    # Include timestamp and level name. Rank will be added manually in messages.
    logging.basicConfig(
        level=log_level_numeric,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,  # Log to standard output
    )

    assert HAVE_RES, "nvidia_resiliency_ext is required for this test."

    # Validate args
    assert (
        args.crash_step > args.local_checkpoint_interval
    ), "Crash step must be after the first local checkpoint interval"
    assert args.crash_step < args.max_steps, "Crash step must be before max_steps"
    assert args.local_checkpoint_interval > 0, "Local checkpoint interval must be positive"

    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if global_rank == 0:
        # Use logger
        logger.info(f"Test started. World size: {world_size}")
        logger.info(f"Using shared log directory: {args.log_dir}")
        logger.info(f"Logging level set to: {args.log_level.upper()}")
        if args.crash_step > 0:
            logger.info(f"Crash simulation enabled at step {args.crash_step}.")
        assert (
            Path(args.log_dir).exists() or args.cleanup_log_dir
        ), f"Log directory {args.log_dir} does not exist and cleanup was not requested."

    # Log Directory Handling (Rank 0 Only)
    log_dir_path = Path(args.log_dir)
    if args.cleanup_log_dir and global_rank == 0:
        if log_dir_path.exists():
            # Use logger
            logger.info(f"[Rank 0] Cleaning up log directory: {log_dir_path}")
            shutil.rmtree(log_dir_path)
        # Use logger
        logger.info(f"[Rank 0] Creating log directory: {log_dir_path}")
        log_dir_path.mkdir(parents=True, exist_ok=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()

    my_local_ckpt_node_dir = get_my_local_ckpt_node_dir(args.log_dir, global_rank)
    logger.info(f"Expecting local checkpoints for this node in: {my_local_ckpt_node_dir}")

    mbs = 1
    gbs = mbs * world_size
    model_config = Llama3Config145M()
    data = MockDataModule(seq_length=model_config.seq_length, global_batch_size=gbs, micro_batch_size=mbs)
    precision_plugin = nl.MegatronMixedPrecision(precision="bf16-mixed")
    strategy = get_megatron_strategy(args, async_save=args.async_save)

    model_run1 = LlamaModel(config=model_config)
    optim_run1 = get_optimizer(bf16_enabled=True)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=args.checkpoint_interval,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
        filename='{model_name}--{val_loss:.2f}-{step}-{consumed_samples}',
    )
    local_checkpoint_callback = res_module.local_checkpoint_callback.LocalCheckpointCallback(
        every_n_train_steps=args.local_checkpoint_interval
    )

    # Instantiate the CrashCallback for Run 1 if crash_step is positive
    crash_callback = CrashCallback(crash_step=args.crash_step if args.crash_step > 0 else None)

    # Define the callback list for Run 1
    # Place LocalCheckpointCallback before CrashCallback
    callbacks_run1 = [
        local_checkpoint_callback,
        crash_callback,
    ]

    nemo_logger_plugin = nl.NeMoLogger(
        log_dir=args.log_dir,
        wandb=None,
        ckpt=checkpoint_callback,
    )

    trainer_run1 = get_trainer(
        args,
        callbacks=callbacks_run1,
        plugins=[precision_plugin],
        strategy=strategy,
    )
    assert hasattr(trainer_run1, 'global_rank'), "Trainer needs global_rank populated."

    logger.info(f"[Rank {global_rank}] Initializing Run 1 Trainer complete. Patching for local I/O.")
    update_trainer_local_checkpoint_io(
        trainer_run1,
        args.log_dir,
        get_global_step_from_global_checkpoint_path,
    )

    resume_logic = nl.AutoResume(resume_if_exists=True, resume_ignore_no_checkpoint=True)
    crashed = False
    # Run 1: Train until Crash
    if global_rank == 0:
        # Use logger
        logger.info("\n" + "=" * 20 + " Starting Run 1: Train until crash " + "=" * 20)

    try:
        llm.train(
            model=model_run1,
            data=data,
            trainer=trainer_run1,
            log=nemo_logger_plugin,
            optim=optim_run1,
            resume=resume_logic,
            tokenizer="data",
        )

        # If crash_step was 0 or None, training should complete Run 1 successfully
        if not args.crash_step or args.crash_step <= 0:
            logger.info(f"[Rank {global_rank}] Run 1 completed successfully (no crash simulated).")

    # Catch the specific exception from the callback
    except CrashException as e:
        logger.info(f"\nSuccessfully caught expected crash: {e}")
        crashed = True

    if torch.distributed.is_initialized():
        dist.barrier()

    latest_local_step_run1 = -1

    logger.info("\nVerifying checkpoints after Run 1...")
    assert crashed, "Training did not crash as expected!"
    logger.info(f"Run 1 finished (crashed at step {args.crash_step}).")

    latest_local_step_run1 = find_latest_local_ckpt_step(my_local_ckpt_node_dir, global_rank)
    assert latest_local_step_run1 is not None, f"No local checkpoints found in {my_local_ckpt_node_dir} after crash!"

    expected_latest_step = (args.crash_step // args.local_checkpoint_interval) * args.local_checkpoint_interval
    logger.info(f"Expected latest local checkpoint step: {expected_latest_step}")
    assert (
        latest_local_step_run1 == expected_latest_step
    ), f"Latest local step {latest_local_step_run1} is not equal to the expected {expected_latest_step}"

    logger.info(f"Latest local checkpoint found at step: {latest_local_step_run1}")

    # --- Run 2: Resume Training ---
    if global_rank == 0:
        # Use logger
        logger.info(
            "\n" + "=" * 20 + f" Starting Run 2: All ranks resuming from step {latest_local_step_run1} " + "=" * 20
        )

    if torch.distributed.is_initialized():
        dist.barrier()

    data = MockDataModule(seq_length=model_config.seq_length, global_batch_size=gbs, micro_batch_size=mbs)
    precision_plugin_run2 = nl.MegatronMixedPrecision(precision="bf16-mixed")
    strategy_run2 = get_megatron_strategy(args, async_save=args.async_save)

    model_run2 = LlamaModel(config=model_config)
    optim_run2 = get_optimizer(bf16_enabled=True)
    data = MockDataModule(seq_length=model_config.seq_length, global_batch_size=gbs, micro_batch_size=mbs)

    checkpoint_callback_run2 = ModelCheckpoint(
        save_last=True,
        monitor="val_loss",
        save_top_k=1,
        every_n_train_steps=args.checkpoint_interval,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
        filename='{model_name}--{val_loss:.2f}-{step}-{consumed_samples}',
    )
    local_checkpoint_callback_run2 = res_module.local_checkpoint_callback.LocalCheckpointCallback(
        every_n_train_steps=args.local_checkpoint_interval
    )
    callbacks_run2 = [local_checkpoint_callback_run2]

    nemo_logger_plugin_run2 = nl.NeMoLogger(
        log_dir=args.log_dir,
        wandb=None,
        ckpt=checkpoint_callback_run2,
    )

    trainer_run2 = get_trainer(
        args,
        callbacks=callbacks_run2,
        plugins=[precision_plugin_run2],
        strategy=strategy_run2,
    )

    assert hasattr(trainer_run2, 'global_rank'), "Trainer needs global_rank populated."
    assert trainer_run2.global_rank == global_rank, "Trainer rank differs from env var rank!"
    assert trainer_run2.world_size == world_size, "Trainer world size differs from env var world size!"

    update_trainer_local_checkpoint_io(
        trainer_run2,
        args.log_dir,
        get_global_step_from_global_checkpoint_path,
    )

    resume_logic_run2 = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=False,
    )
    logger.info(f"[Rank {global_rank}] Starting Run 2 training loop (resuming).")
    llm.train(
        model=model_run2,
        data=data,
        trainer=trainer_run2,
        log=nemo_logger_plugin_run2,
        optim=optim_run2,
        resume=resume_logic_run2,
        tokenizer="data",
    )
    logger.info(f"Run 2 resumed from step {latest_local_step_run1}.")
    latest_local_step_run2 = find_latest_local_ckpt_step(my_local_ckpt_node_dir, global_rank)

    assert (
        latest_local_step_run2 is not None
    ), f"No local checkpoints found in {my_local_ckpt_node_dir} after resuming!"

    expected_latest_step = (args.max_steps // args.local_checkpoint_interval) * args.local_checkpoint_interval
    logger.info(f"Expected latest local checkpoint step: {expected_latest_step}")
    assert (
        latest_local_step_run2 == expected_latest_step
    ), f"Latest local step {latest_local_step_run2} is not equal to the expected {expected_latest_step}"
    logger.info(f"Latest local checkpoint found at step: {latest_local_step_run2}")

    if torch.distributed.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
