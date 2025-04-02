from typing import TypeVar

import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager
from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load_common_state_dict, load_tensors_metadata, load, save
from megatron.core.dist_checkpointing.mapping import ShardedTensor

from nemo.common.plan.plan import Plan
import nemo.lightning as nl
from nemo.lightning.io.pl import TrainerContext
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir


ModelT = TypeVar("ModelT", bound=nn.Module)


class InitMegatronLightningEnv(Plan[nn.Module]):
    def __init__(self, env):
        self.env = env

    def execute(self, model: nn.Module) -> nn.Module:
        if not hasattr(self.env, "trainer"):
            return self._setup(model)

        if isinstance(self.env.trainer.strategy, nl.MegatronStrategy):
            self.env.trainer.strategy.connect(model)
            self.env.fabric.strategy.auto_model_setup = True
            self.env.fabric.strategy.precision.auto_model_setup = True
            self.env.fabric.strategy.dtype_config = (
                self.env.trainer.strategy.precision_plugin.dtype_config
            )
            self.env.fabric.strategy.ddp_config = self.env.trainer.strategy.ddp_config
            self.env.trainer.strategy.megatron_parallel = self._setup(model)._forward_module
        model._fabric = None
        model.trainer = self.env.trainer

        return model

    def _setup(self, model: nn.Module) -> nn.Module:
        return self.env.fabric.setup(model)

    # def extra_repr(self) -> str:
    #     _io = self.fabric.__io__

    #     _accelerator = _io.accelerator
    #     if not self.can_load:
    #         _accelerator = "meta"

    #     out = f"accelerator={_accelerator}, devices={_io.devices}"
    #     # strategy_cfg = _io.strategy
    #     # if hasattr(strategy_cfg, "__fn_or_cls__"):
    #     #     strategy_cfg = strategy_cfg.__fn_or_cls__
    #     # out += f", strategy={strategy_cfg.__name__}()"

    #     if _io.precision:
    #         out += f", precision={_io.precision}"

    #     return out


class InitMegatronModel(Plan):
    def __init__(self, path, env: Plan, path_resolver: str | None = None, convert_state: Plan | None = None):
        super().__init__()
        self._path = path
        self._path_resolver = path_resolver
        if hasattr(env, "trainer"):
            self.trainer = env.trainer
        self.init = InitMegatronLightningEnv(env)

        self.convert_state = convert_state

    def execute(self, model: ModelT) -> ModelT:
        if self.convert_state is not None:
            self.convert_state(model)

        model = self.init(model)
        # Get parameter values before loading
        first_param = next(model.parameters())
        before_shape = first_param.shape
        before_values = first_param[:5].clone().cpu()  # Store on CPU to save GPU memory

        if getattr(model, "_fabric", None):
            print("Loading fabric")
            model._fabric.load(self.path, {"state_dict": model})
        else:
            print(model)

            strategy: nl.MegatronStrategy = self.trainer.strategy
            strategy.trainer = self.trainer

            _before_restore = strategy.restore_config
            strategy.restore_config = RestoreConfig(
                path=self.path, load_artifacts=False
            )
            strategy.selective_restore()
            strategy.restore_config = _before_restore

        # Get parameter values after loading
        first_param = next(model.parameters())
        after_shape = first_param.shape
        after_values = first_param[:5].clone().cpu()  # Store on CPU to save GPU memory

        # Check if values are suspiciously similar
        if before_shape == after_shape:
            rel_diff = torch.abs(after_values - before_values) / (torch.abs(before_values) + 1e-8)
            if torch.mean(rel_diff) < 1e-6:  # Threshold for relative difference
                print("\nWARNING: Parameter values appear almost identical before and after loading.")
                print(f"Mean relative difference: {torch.mean(rel_diff):.2e}")
            else:
                print("Model successfully loaded from: ", self.path)
        else:
            raise ValueError("Model parameters have changed after loading")

        return model

    def extra_repr(self) -> str:
        return f"path={self.path}"

    @property
    def path(self):
        from nemo.common.ckpt.resolver import get_checkpoint

        return get_checkpoint(self._path, self._path_resolver)
    

class MegatronSaveModel(Plan):
    def execute(self, model: ModelT, out_path: str) -> ModelT:
        model.fabric.save(out_path, {"state_dict": model})
        print("Saved to", out_path)

        trainer = nl.Trainer(
            devices=1,
            accelerator="cpu",
            strategy=nl.MegatronStrategy(),
        )
        trainer.strategy._lightning_module = model._forward_module.pipeline
        TrainerContext.from_trainer(trainer).io_dump(
            ckpt_to_context_subdir(out_path), yaml_attrs=["model"]
        )


def save_megatron_state():
    """Save Megatron's parallel state dynamically."""
    state = {}
    for attr_name in dir(parallel_state):
        if attr_name.isupper() and attr_name.startswith('_'):
            state[attr_name] = getattr(parallel_state, attr_name)
    return state


def restore_megatron_state(state):
    """Restore Megatron's parallel state from the saved state."""
    for key, value in state.items():
        setattr(parallel_state, key, value)


@contextmanager
def temporary_single_process_group():
    """
    Context manager to set up a temporary single-process distributed environment.
    """
    original_default_pg = None
    original_megatron_state = save_megatron_state()
    was_initialized = dist.is_initialized()
    single_process_group = None

    try:
        if not was_initialized:
            # Initialize a single-process group if not already initialized
            dist.init_process_group(
                backend='gloo',
                init_method='file:///tmp/tempfile',
                rank=0,
                world_size=1
            )
            single_process_group = dist.group.WORLD
        else:
            # Create a new single-process group if already initialized
            single_process_group = dist.new_group([0], backend='gloo')
            # Temporarily override the default process group
            if hasattr(dist.distributed_c10d, '_world'):
                original_default_pg = dist.distributed_c10d._world.default_pg
                dist.distributed_c10d._world.default_pg = single_process_group

            parallel_state.destroy_model_parallel()

        # Initialize Megatron for single-process operation
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            context_parallel_size=1
        )
        yield single_process_group

    finally:
        # Destroy Megatron's temporary parallel state
        parallel_state.destroy_model_parallel()
        # Restore Megatron's original state
        restore_megatron_state(original_megatron_state)

        # Restore torch.distributed state
        if original_default_pg is not None and hasattr(dist.distributed_c10d, '_world'):
            dist.distributed_c10d._world.default_pg = original_default_pg

        # Clean up the temporary process group
        if not was_initialized and dist.is_initialized():
            dist.destroy_process_group()
        elif was_initialized and single_process_group is not dist.group.WORLD:
            dist.destroy_process_group(single_process_group)


def load_unsharded(checkpoint_dir: str) -> dict:
    """
    Load a distributed checkpoint into an unsharded state dictionary on CPU.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.

    Returns:
        dict: Unsharded state dictionary.
    """
    print(f"Loading checkpoint from {checkpoint_dir}")
    state_dict = {}

    with temporary_single_process_group():
        # Load non-sharded parts of the checkpoint
        common_state = load_common_state_dict(checkpoint_dir)
        print(f"Loaded common state with keys: {list(common_state.keys())}")

        # Load tensor metadata
        metadata = load_tensors_metadata(checkpoint_dir)
        print(f"Found {len(metadata)} tensors in metadata")

        if metadata:
            # Construct ShardedTensor objects for unsharded loading
            sharded_state_dict = {}
            for key, sh_ten_metadata in metadata.items():
                global_shape = sh_ten_metadata.global_shape
                dtype = sh_ten_metadata.dtype
                sharded_tensor = ShardedTensor(
                    key=key,
                    data=None,
                    dtype=dtype,
                    local_shape=global_shape,           # Full tensor loaded locally
                    global_shape=global_shape,
                    global_offset=(0,) * len(global_shape),  # Origin offset
                    axis_fragmentations=(1,) * len(global_shape),  # No sharding
                    replica_id=0,
                    prepend_axis_num=0,
                    allow_shape_mismatch=False
                )
                sharded_state_dict[key] = sharded_tensor

            # Load tensors into the ShardedTensor objects
            loaded_state_dict = load(
                sharded_state_dict,
                checkpoint_dir,
                validate_access_integrity=False
            )
            print(f"Loaded {len(loaded_state_dict)} tensors")

            # Merge loaded tensors with common state
            if "state_dict" in common_state:
                state_dict = common_state["state_dict"]
            else:
                state_dict = common_state
            state_dict.update(loaded_state_dict)
        else:
            state_dict = common_state

    return state_dict


def save_unsharded(state_dict: dict, checkpoint_dir: str, non_sharded_state_dict: dict = None):
    """
    Save a state dictionary to a distributed checkpoint in an unsharded manner.

    Args:
        state_dict (dict): State dictionary containing tensors to be saved unsharded.
        checkpoint_dir (str): Directory where the checkpoint files will be saved.
        non_sharded_state_dict (dict, optional): Dictionary of items to be saved without sharding.
    """
    print(f"Saving unsharded checkpoint to {checkpoint_dir}")

    with temporary_single_process_group():
        # Start with non_sharded_state_dict if provided, otherwise empty dict
        sharded_state_dict = non_sharded_state_dict.copy() if non_sharded_state_dict is not None else {}

        # Process state_dict: wrap tensors in ShardedTensor for unsharded saving
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                global_shape = value.shape
                dtype = value.dtype
                sharded_tensor = ShardedTensor(
                    key=key,
                    data=value,                         # Full tensor data
                    dtype=dtype,
                    local_shape=global_shape,           # Local shape is the full tensor
                    global_shape=global_shape,          # Global shape matches local shape
                    global_offset=(0,) * len(global_shape),  # Offset at origin
                    axis_fragmentations=(1,) * len(global_shape),  # No sharding
                    replica_id=0,                       # Saved by rank 0
                    prepend_axis_num=0,
                    allow_shape_mismatch=False
                )
                sharded_state_dict[key] = sharded_tensor
            else:
                raise ValueError(f"Non-tensor item '{key}' found in state_dict. Please include non-tensors in non_sharded_state_dict.")

        # Save the combined sharded_state_dict using Megatron's save function
        save(
            sharded_state_dict,
            checkpoint_dir,
            validate_access_integrity=False  # No sharding to validate
        )
