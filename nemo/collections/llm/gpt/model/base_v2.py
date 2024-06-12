import functools
import itertools
import os
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sized,
    Tuple,
    Union,
)

import pytorch_lightning as L
import torch
import torch.distributed
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace
from megatron.core.dist_checkpointing.mapping import LocalNonpersitentObject
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from megatron.core.dist_checkpointing.strategies import tensorstore
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer as MCoreTransformerLayer
from megatron.core.utils import (
    get_function_from_registry,
    init_method_normal,
    register_function,
    scaled_init_method_normal,
)
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from torch.optim import Optimizer

from nemo.collections.nlp.models.nlp_model import NLPModel, NLPSaveRestoreConnector
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.transformer import AutocastTransformerLayer, ParallelTransformerLayer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts import utils_funcs
from nemo.core import ModelPT
from nemo.core.config.modelPT import OptimConfig, SchedConfig
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.optim import MainParamsOptimizerWrapper
from nemo.core.optim.optimizers import init_optimizer_states
from nemo.lightning import get_vocab_size, io

# from nemo.lightning.base import ModelConfig
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.utils import AppState, logging
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO
from nemo.utils.model_utils import ckpt_to_dir, inject_model_parallel_rank, uninject_model_parallel_rank

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
# Dataclass util methods


def dataclass_from_dict(klass, d, filter_keys=True):
    try:
        return klass(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except Exception as e:
        raise e
        return d  # Not a dataclass field


def convert_cfg_to_dataclass(cfg, klass):
    model_cfg = OmegaConf.to_object(cfg)
    model_cfg.pop('nemo_version', None)
    model_cfg.pop('target', None)
    data_cls = dataclass_from_dict(klass, model_cfg)
    return data_cls


class LLMSaveRestoreConnector(SaveRestoreConnector):

    def __init__(self):
        super().__init__()
        self.pack_nemo_file = False  # Only save unpacked checkpoint

    def save_to(self, model, save_path: str):
        app_state = AppState()

        dist_ckpt = True
        dist_ckpt_dir = None

        if '.nemo' in save_path and not self.pack_nemo_file:
            dir_name = os.path.dirname(save_path)

        elif '.nemo' in save_path and self.pack_nemo_file:
            dir_name = os.path.dirname(save_path)

        else:
            dir_name = os.path.abspath(os.path.expanduser(save_path))

        # dist ckpt calls save on every rank
        # model weights is a directory
        dist_ckpt_dir = ckpt_to_dir(os.path.join(dir_name, self.model_weights_ckpt))

        # dist checkpoint needs torch.distributed to save the checkpoint
        if not parallel_state.is_initialized():

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

            model.trainer.strategy.setup_megatron_parallel(trainer)
            model.trainer.strategy.setup_precision_plugin()

        # TODO: @Eric - Why does model not have self.sharded_state_dict() anymore?
        sharded_state_dict = model.trainer.strategy.megatron_parallel.sharded_state_dict()

        checkpoint_io = DistributedCheckpointIO(model.cfg.get('dist_ckpt_format', 'zarr'))
        checkpoint_io.save_checkpoint(sharded_state_dict, dist_ckpt_dir)

        # print("dist_ckpt_dir", dist_ckpt_dir)
        # print("dir_name", dir_name)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # create nemo file from folder with all mp_ranks checkpoints
        if (
            app_state.pipeline_model_parallel_rank == 0
            and app_state.tensor_model_parallel_rank == 0
            and app_state.data_parallel_rank == 0
        ):
            with tempfile.TemporaryDirectory() as tmpdir:

                if dist_ckpt:
                    shutil.move(str(dist_ckpt_dir), tmpdir)

                # create config and artifacts in tmpdir
                config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                model.to_config_file(path2yaml_file=config_yaml)
                if hasattr(model, 'artifacts') and model.artifacts is not None:
                    self._handle_artifacts(model, nemo_file_folder=tmpdir)
                    self._update_artifact_paths(model, path2yaml_file=config_yaml)

                # create tar file
                if self.pack_nemo_file:
                    self._make_nemo_file_from_folder(save_path, tmpdir)
                else:
                    # Get the folder path from the save_path and move all values inside the tmpdir to the folder
                    folder_path = dir_name
                    print("folder name", folder_path)

                    for file in os.listdir(tmpdir):
                        shutil.move(os.path.join(tmpdir, file), folder_path)

    def _load_state_dict_from_disk(self, model_weights, map_location=None):
        # if model_weights with the extension removed is a directory, we assume it is a distributed checkpoint
        # we need to defer loading the state dict so we return None
        uninject_model_weights = uninject_model_parallel_rank(model_weights)

        # legacy model_weights will have mp rank injected
        if os.path.isfile(model_weights):
            raise RuntimeError("Non dist checkpoints not supported")
            # return super()._load_state_dict_from_disk(model_weights, map_location)

        # dist checkpoint will be a dir
        elif os.path.isdir(os.path.splitext(uninject_model_weights)[0]):
            return None
        else:
            raise ValueError(f'Expected {model_weights} to be a file or directory.')

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """
        Restores model instance (weights and configuration) into .nemo file

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. By default True
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.

        Example:
            ```
            model = nemo.collections.nlp.models.TextClassification.restore_from('asr.nemo')
            assert isinstance(model, nemo.collections.nlp.models.TextClassification)
            ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """

        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        loaded_params = super().load_config_and_state_dict(
            calling_cls,
            restore_path,
            override_config_path,
            map_location,
            strict,
            return_config,
            trainer,
        )
        if not isinstance(loaded_params, tuple) or return_config is True:
            return loaded_params
        conf, instance, state_dict = loaded_params

        # if we're using dist checkpointing then state_dict will be None
        # if state_dict is None:
        # dist checkpointing needs torch.distributed to load the checkpoint
        if not parallel_state.is_initialized():

            def dummy():
                return

            if trainer.strategy.launcher is not None:
                trainer.strategy.launcher.launch(dummy, trainer=trainer)
            trainer.strategy.setup_environment()

        trainer.strategy.model = instance
        trainer.strategy.setup_megatron_parallel(trainer)
        trainer.strategy.setup_precision_plugin()

        instance.configure_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Check if self.model_extracted_dir is set, and is a valid path
            if self.model_extracted_dir is not None and os.path.isdir(self.model_extracted_dir):
                # Log that NeMo will use the provided `model_extracted_dir`
                logging.info(
                    f"Restoration will occur within pre-extracted directory : " f"`{self.model_extracted_dir}`."
                )

                # Override `tmpdir` above with the pre-extracted `model_extracted_dir`
                tmpdir = self.model_extracted_dir

            else:
                # Extract the nemo file into the temporary directory
                self._unpack_nemo_file(
                    path2file=restore_path, out_folder=tmpdir, extract_config_only=return_config is True
                )
            checkpoint = {}

            # TODO: @Eric, why does model no longer have sharded state dict?
            sharded_state_dict = instance.trainer.strategy.megatron_parallel.sharded_state_dict()
            checkpoint['state_dict'] = sharded_state_dict

            # remove model weights extension
            tmp_model_weights_ckpt = os.path.join(tmpdir, self.model_weights_ckpt)
            tmp_model_weights_dir = os.path.splitext(tmp_model_weights_ckpt)[0]
            assert os.path.isdir(tmp_model_weights_dir), f'Expected {tmp_model_weights_dir} to be a directory.'
            checkpoint_io = DistributedCheckpointIO.from_config(conf)
            checkpoint = checkpoint_io.load_checkpoint(
                tmp_model_weights_dir, sharded_state_dict=checkpoint, strict=strict
            )
            instance.on_load_checkpoint(checkpoint)
            if hasattr(instance, 'setup_transformer_engine_tp_groups'):
                instance.setup_transformer_engine_tp_groups()

        # else:
        #     state_dict = self.modify_state_dict(conf, state_dict)
        #     super().load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
        return instance


@dataclass
class GPTOptimConfig(OptimConfig):
    name: str = "fused_adam"
    lr: float = 1e-4
    weight_decay: float = 0.0

    sched: Optional[SchedConfig] = None


@dataclass
class GPTConfigV2(TransformerConfig):
    # From megatron.core.models.gpt.gpt_model.GPTModel

    """Configuration object for megatron-core transformers.

    The initialization function has an argument for each parameter, including those in ModelParallelConfig.
    """

    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: str = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024

    optim: OptimConfig = OptimConfig(name='fused_adam')
    optimizer_fn: Optional[str] = None

    tokenizer_filepath: Optional[str] = None

    def configure_model(self, tokenizer) -> "MCoreGPTModel":
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
        from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

        return MCoreGPTModel(
            self,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        )

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        super().__post_init__()

        # Register a default function for initialization and restoration
        init_method_fn = init_method_normal(self.init_method_std)
        register_function(init_method_fn)

        if self.init_method is None:
            self.init_method = init_method_fn.__name__  # init_method_normal(self.init_method_std)

        # Register a default function for initialization and restoration
        scaled_init_method_normal_fn = scaled_init_method_normal(self.init_method_std, self.num_layers)
        register_function(scaled_init_method_normal_fn)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal_fn.__name__


class GPTModelV2(
    ModelPT,  # NLPModel
    # io.IOMixin,
    # io.ConnectorMixin
):
    def __init__(
        self,
        cfg: GPTConfigV2,
        trainer: L.Trainer = None,
    ):
        # If function registrations are part of dataclass, must come first, but can be done outside dataclass to put this after __init__
        if not isinstance(cfg, GPTConfigV2):
            cfg = convert_cfg_to_dataclass(cfg, GPTConfigV2)

        # ModelPT init
        super().__init__(cfg, trainer=trainer)

        # Dataclass config here; OmegaConf config is stored under self.cfg
        self.mcore_config = cfg

        # Handle tokenizer
        tokenizer_filepath = self.cfg.get("tokenizer_filepath", None)
        if tokenizer_filepath is not None:
            tokenizer_filepath = self.register_artifact('tokenizer_filepath', tokenizer_filepath)
            self.tokenizer = get_nmt_tokenizer(tokenizer_model=tokenizer_filepath)
        else:
            # Use default tokenizer
            self.tokenizer = get_nmt_tokenizer("megatron", "GPT2BPETokenizer")

    def configure_model(self) -> None:
        if not hasattr(self, 'module'):  # ptl demands this function be a no-op after first call
            self.module = self.mcore_config.configure_model(self.tokenizer)

    def configure_optimizers(self) -> Optimizer:
        if self.mcore_config.optimizer_fn is not None:
            optimizer_fn = get_function_from_registry(self.mcore_config.optimizer_fn)
            return optimizer_fn(self)

        return super().configure_optimizers()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return gpt_data_step(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return gpt_forward_step(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        return MaskedTokenLossReduction()

    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        return MaskedTokenLossReduction(validation_step=True)

    def copy(self) -> "GPTModel":
        return self.__class__(self.config, self.trainer)

    # Can be aliased to support config based dataloaders (1.x style for backward compat)
    def setup_training_data(self, cfg):
        pass

    def setup_validation_data(self, cfg):
        pass

    def setup_test_data(self, cfg):
        pass

    # Put NGC models here
    def list_available_models(cls):
        return []

    # This enables models to restore from without providing the connector explicitly
    @classmethod
    def get_default_save_restore_connector(cls):
        return LLMSaveRestoreConnector()


def gpt_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask"))
    # if self.get_attention_mask_from_fusion:
    #     required_keys.remove('attention_mask')

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def gpt_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


def gpt_default_optimizer(module) -> Optimizer:
    from apex.optimizers import FusedAdam

    return FusedAdam(module.parameters(), lr=1e-4)


def get_batch_on_this_context_parallel_rank(batch):
    from megatron.core import parallel_state

    if cp_size := parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if 'loss_mask' in batch and batch['loss_mask'] is not None:
            num_valid_tokens_in_ub = batch['loss_mask'].sum()

        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != 'attention_mask' else 2
                _val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                    non_blocking=True
                )
                _val = _val.index_select(seq_dim, index)
                _val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
                batch[key] = _val
        batch['num_valid_tokens_in_ub'] = num_valid_tokens_in_ub
    return batch


def get_packed_seq_params(batch):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if cu_seqlens_argmin := batch.get('cu_seqlens_argmin', None) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch['max_seqlen'].squeeze() if 'max_seqlen' in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )


__all__ = ["GPTModel", "GPTConfig", "gpt_data_step", "gpt_forward_step", "gpt_default_optimizer"]
