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

import contextlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from megatron.core.num_microbatches_calculator import update_num_microbatches
from torch.nn import Module

from nemo.automodel.config import ConfigContainer
from nemo.automodel.utils.common_utils import unwrap_model
from nemo.tron.checkpointing import (
    TRACKER_PREFIX,
    checkpoint_exists,
    get_checkpoint_run_config_filename,
    get_checkpoint_train_state_filename,
    read_run_config,
    read_train_state,
)
from nemo.tron.checkpointing import (
    TRAIN_STATE_FILE as _TRON_TRAIN_STATE_FILE,
)
from nemo.tron.state import GlobalState, TrainState
from nemo.tron.utils import wandb_utils
from nemo.tron.utils.common_utils import (
    get_rank_safe,
    get_world_size_safe,
    print_rank_0,
)

# -----------------------------------------------------------------------------
# Filenames & helpers
# -----------------------------------------------------------------------------

MODEL_WEIGHTS_FILE = "model.pt"
TRAINER_STATE_FILE = "trainer.pt"

# Alias for consistency with Tron naming.
TRAIN_STATE_FILE: str = _TRON_TRAIN_STATE_FILE  # "train_state.pt"


# Adapted from https://github.com/huggingface/transformers/pull/34632
def safe_globals():
    from packaging import version

    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


# -----------------------------------------------------------------------------
# Internal helpers (RNG state)
# -----------------------------------------------------------------------------


def _collect_rng_state() -> Dict[str, Any]:
    """Capture Python / NumPy / Torch RNG states for reproducibility."""

    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }


def _apply_rng_state(rng_state: Dict[str, Any]):  # pragma: no cover
    """Restore RNG state collected by :func:`_collect_rng_state`."""

    random.setstate(rng_state["random_rng_state"])
    np.random.set_state(rng_state["np_rng_state"])
    torch.set_rng_state(rng_state["torch_rng_state"])
    torch.cuda.set_rng_state_all(rng_state["cuda_rng_state"])


# Logger
logger = logging.getLogger(__name__)


def get_checkpoint_name(
    checkpoints_path,
    iteration,
    release=False,
):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    common_path = os.path.join(checkpoints_path, directory)
    return common_path


def save_checkpoint(
    save_dir: str | Path,
    state: GlobalState,
    model: Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    save_rng: bool = True,
    save_optim: bool = True,
) -> None:
    """Save a distributed checkpoint.

    Rank‑0 performs the actual I/O. Any exception raised on rank‑0 is
    broadcast to all ranks and re‑raised so that the entire job fails
    coherently instead of dead‑locking on later collectives.
    """

    import torch.distributed as dist  # local import to avoid circular deps during unit tests

    rank = get_rank_safe()
    world_size = get_world_size_safe()

    caught_exc: Optional[BaseException] = None

    if rank == 0:
        try:
            # ----------------------------------------------------------------------------------
            # Existing save logic – unchanged except for indentation so it lives in the try‑block
            # ----------------------------------------------------------------------------------
            save_dir = Path(save_dir)

            # layout: <save_dir>/iter_XXXXXX/
            iteration: int = state.train_state.step
            ckpt_dir = Path(get_checkpoint_name(str(save_dir), iteration))
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # 1. Model weights ---------------------------------------------------------------
            model = unwrap_model(model)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(ckpt_dir)
            else:
                model_path = ckpt_dir / MODEL_WEIGHTS_FILE
                torch.save(model.state_dict(), model_path)

            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(ckpt_dir)
            elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "save_pretrained"):
                tokenizer._tokenizer.save_pretrained(ckpt_dir)

            # 2. Trainer state --------------------------------------------------------------
            trainer_state: dict[str, Any] = {}
            if save_optim and optimizer is not None:
                trainer_state["optimizer"] = optimizer.state_dict()
            if save_optim and scheduler is not None and hasattr(scheduler, "state_dict"):
                trainer_state["scheduler"] = scheduler.state_dict()
            if save_rng:
                trainer_state["rng_state"] = _collect_rng_state()

            trainer_path = ckpt_dir / TRAINER_STATE_FILE
            torch.save(trainer_state, trainer_path)

            # 3. Run config -----------------------------------------------------------------
            cfg_path = Path(get_checkpoint_run_config_filename(str(ckpt_dir)))
            state.cfg.to_yaml(str(cfg_path))  # type: ignore[attr-defined]

            # 4. TrainState tracker files ---------------------------------------------------
            ts_dict = state.train_state.state_dict()
            ts_local = Path(get_checkpoint_train_state_filename(str(ckpt_dir)))
            ts_global = Path(get_checkpoint_train_state_filename(str(save_dir), prefix=TRACKER_PREFIX))

            torch.save(ts_dict, ts_local)
            torch.save(ts_dict, ts_global)

            # WandB artifact ---------------------------------------------------------------
            if state.wandb_logger:
                wandb_utils.on_save_checkpoint_success(
                    str(ckpt_dir),
                    str(save_dir),
                    iteration,
                    wandb_writer=state.wandb_logger,
                )

            print_rank_0(
                f"[Checkpoint] Finished writing checkpoint to {ckpt_dir} (world size={get_world_size_safe()})",
            )
        except BaseException as exc:
            # Capture any exception so we can broadcast it.
            caught_exc = exc

    # --------------------------------------------------------------------------------------
    # Synchronise across ranks and propagate any exception raised on rank‑0.
    # --------------------------------------------------------------------------------------
    if dist.is_initialized() and world_size > 1:
        # NOTE: We wrap the exception in a list because broadcast_object_list wants a list.
        exc_container = [caught_exc]
        dist.broadcast_object_list(exc_container, src=0)
        caught_exc = exc_container[0]

    # Re‑raise on *all* ranks if something went wrong.
    if caught_exc is not None:
        raise caught_exc


def get_base_checkpoint_dir(
    load_dir: str | Path, pretrained_dir: Optional[str | Path] = None
) -> tuple[str | Path, bool]:
    # Finetuning directories
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(
            f"Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}"
        )
        if not checkpoint_exists(pretrained_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        return pretrained_dir, True

    return load_dir, False


def load_checkpoint(
    state: GlobalState,
    model: Optional[Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[Dict[str, Any], TrainState]:
    load_dir = state.cfg.checkpoint_config.load
    pretrained_dir = state.cfg.checkpoint_config.pretrained_checkpoint
    load_dir, finetune = get_base_checkpoint_dir(load_dir, pretrained_dir)
    root_dir = Path(load_dir)

    if not checkpoint_exists(str(root_dir)):
        raise FileNotFoundError(f"No checkpoint metadata found in {root_dir}")

    # Determine latest iteration via tracker file
    train_state_tracker = Path(get_checkpoint_train_state_filename(str(root_dir), prefix=TRACKER_PREFIX))
    tracker_ts = read_train_state(str(train_state_tracker))
    iteration: int = tracker_ts.step

    ckpt_dir = Path(get_checkpoint_name(str(root_dir), iteration))
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found")

    # 2. Model --------------------------------------------------------------------------
    if model is not None:
        model = unwrap_model(model)
        if hasattr(model, "from_pretrained"):
            model.from_pretrained(ckpt_dir)
        else:
            model_path = ckpt_dir / MODEL_WEIGHTS_FILE
            if not model_path.exists():
                raise FileNotFoundError(f"Model weights not found at {model_path}")
            state_dict = torch.load(model_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logger.warning("[Checkpoint] Model load - missing: %s  unexpected: %s", missing, unexpected)

    if not finetune:
        trainer_path = ckpt_dir / TRAINER_STATE_FILE
        if not trainer_path.exists():
            raise FileNotFoundError(f"Trainer state not found at {trainer_path}")

        with safe_globals():
            trainer_state_loaded: Dict[str, Any] = torch.load(trainer_path, map_location="cpu")

        # 3. Optimiser / scheduler ----------------------------------------------------------
        if optimizer is not None and "optimizer" in trainer_state_loaded:
            optimizer.load_state_dict(trainer_state_loaded["optimizer"])
        if scheduler is not None and "scheduler" in trainer_state_loaded:
            scheduler.load_state_dict(trainer_state_loaded["scheduler"])

        # 4. RNG -----------------------------------------------------------------------------
        if "rng_state" in trainer_state_loaded:
            _apply_rng_state(trainer_state_loaded["rng_state"])

        # 5. TrainState ---------------------------------------------------------------------
        state.train_state = read_train_state(get_checkpoint_train_state_filename(str(ckpt_dir)))

        # 1. Run config ---------------------------------------------------------------------
        cfg_path = Path(get_checkpoint_run_config_filename(str(ckpt_dir)))
        config = read_run_config(str(cfg_path))

        # Update number of microbatches calculator based on loaded samples
        update_num_microbatches(state.train_state.consumed_train_samples, verbose=True)

        # Optionally override cfg if not already set
        if config is not None and state.cfg is None:
            state.cfg = config if isinstance(config, ConfigContainer) else state.cfg

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f"  successfully loaded checkpoint from {load_dir} "
        # f"[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, "
        # f"p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ] "
        f"at iteration {state.train_state.step}"
    )

    torch.cuda.empty_cache()
    # WandB artifact usage logging
    if state.wandb_logger and (
        not torch.distributed.is_initialized() or get_rank_safe() == (get_world_size_safe() - 1)
    ):
        wandb_utils.on_load_checkpoint_success(str(ckpt_dir), str(root_dir), state.wandb_logger)
