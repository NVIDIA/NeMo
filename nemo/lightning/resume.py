import os
from dataclasses import dataclass
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Union

import lightning_fabric as fl
import pytorch_lightning as pl

from nemo.lightning import io
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import uninject_model_parallel_rank

# Dynamically inherit from the correct Path subclass based on the operating system.
if os.name == "nt":
    BasePath = WindowsPath
else:
    BasePath = PosixPath


@dataclass(kw_only=True)
class AutoResume:
    """Class that handles the logic for setting checkpoint paths and restoring from
    checkpoints in NeMo.

    Attributes:
        restore_path (str): Path to restore model from. If importing a checkpoint from HF or
            another non-NeMo checkpoint format, the checkpoint will be automatically converted to a NeMo compatible format.
            resume_path or the run's log_dir takes precedence over restore_path.
        resume_path (str): Path to the checkpointing directory to restore from. Defaults to <log_dir>/checkpoints
        adapter_path (str): Path to any adapter checkpoints.
        resume_if_exists (bool): Whether this experiment is resuming from a previous run. If
            True, it sets trainer._checkpoint_connector._ckpt_path so that the trainer should
            auto-resume. exp_manager will move files under log_dir to log_dir/run_{int}.
            Defaults to False.
        resume_past_end (bool): By default, AutoResume throws an error if resume_if_exists is
            True and a checkpoint matching ``*end.ckpt`` indicating a previous training run
            fully completed. Setting resume_past_end=True disables this behavior and loads the
            last checkpoint.
        resume_ignore_no_checkpoint (bool): AutoResume throws an error if resume_if_exists is
            True and no checkpoint could be found. Setting resume_ignore_no_checkpoint=True
            disables this behavior, in which case exp_manager will print a message and
            continue without restoring.
        restore_optimizer_states (bool): Flag to control whether to restore optimizer states as well.
            This has no effect if restore_path is None.
    """

    restore_path: Optional[str] = None
    resume_path: Optional[str] = None
    adapter_path: Optional[str] = None
    resume_if_exists: bool = False
    resume_past_end: bool = False
    resume_ignore_no_checkpoint: bool = False
    restore_optimizer_states: bool = False

    def setup(self, trainer: Union[pl.Trainer, fl.Fabric], model=None):
        if isinstance(trainer, fl.Fabric):
            raise NotImplementedError("Fabric is not supported yet.")

        ckpt_path, restore_path = self.nemo_path(model)
        if ckpt_path:
            trainer.ckpt_path = ckpt_path
            trainer.checkpoint_callback.last_model_path = ckpt_path
        elif restore_path:
            trainer.strategy.restore_path = restore_path
            trainer.strategy.restore_optimizer_states = self.restore_optimizer_states

    def nemo_path(self, model: Optional[io.ConnectorMixin] = None) -> tuple[Optional[Path], Optional[Path]]:
        from nemo.utils.exp_manager import NotFoundError, _filter_out_unfinished_checkpoints

        ### refactored from exp_manager
        checkpoint = None
        app_state = AppState()
        log_dir = app_state.log_dir
        app_state.restore = self.resume_if_exists
        if self.resume_if_exists:
            # Use <log_dir>/checkpoints/ unless `dirpath` is set
            checkpoint_dir = Path(self.resume_path) if self.resume_path else Path(Path(log_dir) / "checkpoints")

            # when using distributed checkpointing, checkpoint_dir is a directory of directories
            # we check for this here
            dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
            end_dist_checkpoints = [d for d in dist_checkpoints if d.match("*end")]
            last_dist_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

            end_chkpt_cnt = len(end_dist_checkpoints)
            end_checkpoints = _filter_out_unfinished_checkpoints(end_dist_checkpoints)
            finished_end_chkpt_cnt = len(end_checkpoints)
            if end_chkpt_cnt > 0 and finished_end_chkpt_cnt == 0:
                raise ValueError(
                    "End checkpoint is unfinished and cannot be used to resume the training."
                    " Please remove the checkpoint manually to avoid unexpected cosequences, such as"
                    " restarting from scratch."
                )

            last_chkpt_cnt = len(last_dist_checkpoints)
            last_checkpoints = _filter_out_unfinished_checkpoints(last_dist_checkpoints)
            finished_last_chkpt_cnt = len(last_checkpoints)
            if last_chkpt_cnt > 0 and finished_last_chkpt_cnt == 0:
                raise ValueError(
                    "Last checkpoint is unfinished and cannot be used to resume the training."
                    " Please remove the checkpoint manually to avoid unexpected cosequences, such as"
                    " restarting from scratch. Hint: Iteration number can be added to the checkpoint name pattern"
                    " to maximize chance that there is at least one finished last checkpoint to resume from."
                )

            if not checkpoint_dir.exists() or (not len(end_checkpoints) > 0 and not len(last_checkpoints) > 0):
                if self.resume_ignore_no_checkpoint:
                    warn = f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. "
                    if checkpoint is None:
                        warn += "Training from scratch."
                    logging.warning(warn)
                else:
                    raise NotFoundError(
                        f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume."
                    )
            elif len(end_checkpoints) > 0:
                if self.resume_past_end:
                    if len(end_checkpoints) > 1:
                        if "mp_rank" in str(end_checkpoints[0]):
                            checkpoint = end_checkpoints[0]
                        else:
                            raise ValueError(f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
                else:
                    raise ValueError(
                        f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
                    )
            elif len(last_checkpoints) > 1:
                if any([s for s in ["mp_rank", "tp_rank", "fsdp_shard"] if s in str(last_checkpoints[0])]):
                    checkpoint = last_checkpoints[0]
                    checkpoint = uninject_model_parallel_rank(checkpoint)
                else:
                    # Select the checkpoint with the latest modified time
                    checkpoint = sorted(last_checkpoints, key=lambda pth: pth.lstat().st_mtime, reverse=True)[0]
                    logging.warning(
                        f"Multiple checkpoints {last_checkpoints} matches *last.ckpt. Selecting one with the latest modified time."
                    )
            else:
                checkpoint = last_checkpoints[0]

        if not checkpoint:
            if self.restore_path:
                try:
                    restore_path = model.import_ckpt(self.restore_path)
                except (ValueError, AttributeError):
                    restore_path = self.restore_path

                if self.adapter_path:
                    restore_path = AdapterPath(restore_path, adapter_path=Path(self.adapter_path))

                if isinstance(restore_path, str):
                    restore_path = Path(restore_path)

                return None, restore_path

        if checkpoint:
            if self.adapter_path:
                return AdapterPath(checkpoint, adapter_path=Path(self.adapter_path)), None
            return Path(checkpoint), None

        return None, None


class AdapterPath(BasePath):
    adapter_path: Optional[Path]

    def __new__(cls, *args, adapter_path: Optional[Path] = None, **kwargs):
        output = super().__new__(cls, *args, **kwargs)
        output.adapter_path = adapter_path
        return output
