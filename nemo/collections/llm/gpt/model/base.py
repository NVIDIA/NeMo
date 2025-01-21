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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Union

import lightning.pytorch as L
import torch
import torch.distributed
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch import nn

from nemo.collections.llm import fn
from nemo.lightning import get_vocab_size, io
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule, OptimizerModule
from nemo.utils import logging
from nemo.utils.import_utils import safe_import_from

TransformerLayer, HAVE_TE = safe_import_from("transformer_engine.pytorch", "TransformerLayer")

# Gradient accumulation fusion may be enabled if available, for more information see:
# https://github.com/NVIDIA/Megatron-LM/blob/01945b98d1ea3a2acb5e8301e181a328104f4856/megatron/core/tensor_parallel/layers.py#L575
# TODO: Clean this up with a getter and install instructions
_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


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

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    if 'cu_seqlens' in _batch:
        required_device_keys.add('cu_seqlens')
        required_host_keys.add('cu_seqlens_argmin')
        required_host_keys.add('max_seqlen')

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch_required_keys)

    return output


def gpt_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
    }

    if 'attention_mask' not in batch:
        assert (
            HAVE_TE
        ), "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
            This requires Transformer Engine's implementation of fused or flash attention."
    else:
        forward_args["attention_mask"] = batch['attention_mask']

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


def transformer_engine_layer_spec(config: "GPTConfig") -> ModuleSpec:
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        fp8=bool(config.num_moe_experts and (config.fp8 is not None)),
    )


def transformer_engine_full_layer_spec(config: "GPTConfig") -> ModuleSpec:
    assert HAVE_TE, "Please ensure Megatron Core and Transformer Engine are installed."
    num_layers = get_num_layers_to_build(config)
    return TransformerBlockSubmodules(
        layer_specs=[ModuleSpec(module=TETransformerLayerAutocast)] * num_layers, layer_norm=FusedLayerNorm
    )


def local_layer_spec(config: "GPTConfig") -> ModuleSpec:
    from megatron.core.models.gpt import gpt_layer_specs

    return gpt_layer_specs.get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm, qk_layernorm=config.qk_layernorm
    )


def default_layer_spec(config: "GPTConfig") -> ModuleSpec:
    if HAVE_TE:
        if config.use_transformer_engine_full_layer_spec:
            return transformer_engine_full_layer_spec(config)
        else:
            return transformer_engine_layer_spec(config)
    else:
        return local_layer_spec(config)


def torch_dtype_from_mcore_config(config: TransformerConfig):
    if config.fp16:
        return torch.float16
    elif config.bf16:
        return torch.bfloat16
    else:
        return torch.float


@dataclass
class GPTConfig(TransformerConfig, io.IOMixin):
    # From megatron.core.models.gpt.gpt_model.GPTModel
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True
    make_vocab_size_divisible_by: int = 128
    position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    seq_len_interpolation_factor: Optional[float] = None
    seq_length: int = 1024
    attention_softmax_in_fp32: bool = False
    masked_softmax_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    gradient_accumulation_fusion: bool = _grad_accum_fusion_available
    deallocate_pipeline_outputs: bool = True
    scatter_embedding_sequence_parallel: bool = True
    tp_only_amax_red: bool = False

    use_transformer_engine_full_layer_spec: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTConfig"], ModuleSpec]] = default_layer_spec

    forward_step_fn: Callable = gpt_forward_step
    data_step_fn: Callable = gpt_data_step

    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "MCoreGPTModel":
        if self.enable_cuda_graph:
            assert HAVE_TE, "Transformer Engine is required for cudagraphs."
            assert getattr(self, 'use_te_rng_tracker', False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)

        if hasattr(self, 'vocab_size'):
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

        model = MCoreGPTModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
        )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if HAVE_TE and self.use_transformer_engine_full_layer_spec:
            # Copied from: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_tensor_parallel_group"):
                        tp_group = parallel_state.get_tensor_model_parallel_group()
                        child.set_tensor_parallel_group(tp_group)

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_stream = torch.cuda.Stream()
                for module in self.get_model_module_list():
                    for index, child in enumerate(module.modules()):
                        if index == 0:
                            continue
                        if hasattr(child, "set_context_parallel_group"):
                            child.set_context_parallel_group(
                                parallel_state.get_context_parallel_group(),
                                parallel_state.get_context_parallel_global_ranks(),
                                cp_stream,
                            )

        return model


@dataclass
class GPTConfig126M(GPTConfig):
    seq_length: int = 2048
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig5B(GPTConfig):
    seq_length: int = 2048
    num_layers: int = 24
    hidden_size: int = 4096
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig7B(GPTConfig):
    seq_length: int = 2048
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 10880
    num_attention_heads: int = 32
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig20B(GPTConfig):
    seq_length: int = 2048
    num_layers: int = 44
    hidden_size: int = 6144
    ffn_hidden_size: int = 24576
    num_attention_heads: int = 48
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig40B(GPTConfig):
    seq_length: int = 2048
    num_layers: int = 48
    hidden_size: int = 8192
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 64
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True


@dataclass
class GPTConfig175B(GPTConfig):
    seq_length: int = 2048
    num_layers: int = 96
    hidden_size: int = 12288
    ffn_hidden_size: int = 49152
    num_attention_heads: int = 96
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    use_transformer_engine_full_layer_spec: bool = True
    layernorm_zero_centered_gamma: bool = True


class GPTModel(L.LightningModule, io.IOMixin, io.ConnectorMixin, fn.FNMixin):
    def __init__(
        self,
        config: GPTConfig,
        # TODO: Add transformer_layer_spec when we update mcore
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.optim = optim or MegatronOptimizerModule(config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=True))
        self.optim.connect(self)  # This will bind the `configure_optimizers` method
        self.model_transform = model_transform
        self._training_loss_reduction = None
        self._validation_loss_reduction = None

    def configure_model(self) -> None:
        if not hasattr(self, "module"):
            self.module = self.config.configure_model(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        extra_kwargs = {'packed_seq_params': packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            **extra_kwargs,
        )

        return output_tensor

    def data_step(self, dataloader_iter) -> Dict[str, torch.Tensor]:
        return self.config.data_step_fn(dataloader_iter)

    def forward_step(self, batch) -> torch.Tensor:
        return self.config.forward_step_fn(self, batch)

    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        return self.forward_step(batch)

    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)

    def get_inference_wrapper(self, params_dtype, inference_batch_times_seqlen_threshold) -> torch.Tensor:
        # This is to get the MCore model required in GPTInferenceWrapper.
        mcore_model = self.module
        while mcore_model:
            if type(mcore_model) is MCoreGPTModel:
                break
            mcore_model = getattr(mcore_model, "module", None)
        if mcore_model is None or type(mcore_model) is not MCoreGPTModel:
            raise ValueError("Exact McoreGPTModel instance not found in the model structure.")

        vocab_size = None
        if self.tokenizer is not None:
            vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.config, 'vocab_size'):
            vocab_size = self.config.vocab_size
        else:
            raise ValueError(
                'Unable to find vocab size. Either pass in a tokenizer with vocab size, or set vocab size in the model config'
            )

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=mcore_model.config.hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=vocab_size,
        )

        model_inference_wrapper = GPTInferenceWrapper(mcore_model, inference_wrapper_config)
        return model_inference_wrapper

    @property
    def training_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self) -> MaskedTokenLossReduction:
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True)

        return self._validation_loss_reduction


class AutocastTransformerLayer(TransformerLayer):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float,
        num_attention_heads: int,
        init_method: Callable,
        output_layer_init_method: Callable,
        hidden_dropout: float,
        attention_dropout: float,
        layer_number: Optional[int] = None,
        kv_channels: Optional[int] = None,
        self_attn_mask_type: str = "causal",
        tp_group: Optional[Any] = None,
        tp_size: int = 1,
        params_dtype: torch.dtype = torch.float32,
        get_rng_state_tracker: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        sequence_parallel: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0,
        use_emha: bool = False,
        ub_tp_comm_overlap: bool = False,
        ub_bulk_wgrad: bool = True,
        ub_bulk_dgrad: bool = True,
        autocast_dtype: Any = 16,
        zero_centered_gamma: bool = False,
        device: str = 'cuda',
        **kwargs,
    ) -> None:
        assert HAVE_TE, "AutocastTransformerLayer requires Megatron Core and Transformer Engine to be installed."

        transformer_layer_args = {
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "layernorm_epsilon": layernorm_epsilon,
            "num_attention_heads": num_attention_heads,
            "init_method": init_method,
            "output_layer_init_method": output_layer_init_method,
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
            "layer_number": layer_number,
            "kv_channels": kv_channels,
            "self_attn_mask_type": self_attn_mask_type,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "params_dtype": params_dtype,
            "get_rng_state_tracker": get_rng_state_tracker,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "seq_length": seq_length,
            "micro_batch_size": micro_batch_size,
            "sequence_parallel": sequence_parallel,
            "apply_residual_connection_post_layernorm": apply_residual_connection_post_layernorm,
            "output_layernorm": output_layernorm,
            "layer_type": layer_type,
            "drop_path_rate": drop_path_rate,
            "set_parallel_mode": tp_size > 1,
            "fuse_qkv_params": True,
            "zero_centered_gamma": zero_centered_gamma,
            "ub_tp_comm_overlap": ub_tp_comm_overlap,
            "ub_bulk_wgrad": ub_bulk_wgrad,
            "ub_bulk_dgrad": ub_bulk_dgrad,
            "device": device,
        }
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version > packaging.version.Version("1.5.0"):
            for comm in ["ag", "rs"]:
                ub_overlap_flag = "ub_overlap_" + comm
                split_gemm_flag = "ub_split_" + comm
                atomic_gemm_flag = "ub_atomic_gemm_" + comm
                # Use old overlap flags if they were supplied instead
                if ub_overlap_flag in kwargs:
                    transformer_layer_args[ub_overlap_flag] = kwargs[ub_overlap_flag]
                else:
                    transformer_layer_args[ub_overlap_flag] = kwargs.get(split_gemm_flag, True) or kwargs.get(
                        atomic_gemm_flag, False
                    )
            if te_version > packaging.version.Version("1.6.0.dev0"):
                transformer_layer_args["ub_overlap_rs_dgrad"] = kwargs.get("ub_overlap_rs_dgrad", False)
        else:
            transformer_layer_args["ub_split_ag"] = kwargs.get("ub_split_ag", True)
            transformer_layer_args["ub_split_rs"] = kwargs.get("ub_split_rs", True)
            transformer_layer_args["ub_atomic_gemm_ag"] = kwargs.get("ub_atomic_gemm_ag", False)
            transformer_layer_args["ub_atomic_gemm_rs"] = kwargs.get("ub_atomic_gemm_rs", False)
        super().__init__(**transformer_layer_args)

        # Dtype for forward pass - ignore amp O2
        self.dtype = utils_funcs.torch_dtype_from_precision(autocast_dtype, megatron_amp_O2=None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_output: Optional[torch.Tensor] = None,
        enc_dec_attn_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        if self.dtype == torch.float32:
            return super().forward(
                hidden_states,
                attention_mask,
                encoder_output=encoder_output,
                enc_dec_attn_mask=enc_dec_attn_mask,
                inference_params=inference_params,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
            )
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                encoder_output=encoder_output,
                enc_dec_attn_mask=enc_dec_attn_mask,
                inference_params=inference_params,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
            )


class TETransformerLayerAutocast(AutocastTransformerLayer, BaseTransformerLayer):
    def __init__(self, config, layer_number=1, hidden_dropout=None):
        assert HAVE_TE, "TETransformerLayerAutocast requires Megatron Core and Transformer Engine to be installed."

        self.config = config
        self.is_first_microbatch = True
        precision = 'bf16' if config.bf16 else 16

        transformer_layer_args = {
            "hidden_size": config.hidden_size,
            "ffn_hidden_size": config.ffn_hidden_size,
            "layernorm_epsilon": config.layernorm_epsilon,
            "num_attention_heads": config.num_attention_heads,
            "init_method": config.init_method,
            "output_layer_init_method": config.output_layer_init_method,
            "hidden_dropout": config.hidden_dropout,
            "attention_dropout": config.attention_dropout,
            "layer_number": layer_number + self._get_layer_offset(),
            "kv_channels": config.kv_channels,
            "tp_size": parallel_state.get_tensor_model_parallel_world_size(),
            "params_dtype": config.params_dtype,
            "get_rng_state_tracker": tensor_parallel.random.get_cuda_rng_tracker,
            "fuse_wgrad_accumulation": config.gradient_accumulation_fusion,
            "seq_length": None,  # used for jit warmup
            "micro_batch_size": None,  # used for jit warmup
            "sequence_parallel": config.sequence_parallel,
            "apply_residual_connection_post_layernorm": config.apply_residual_connection_post_layernorm,
            "autocast_dtype": precision,
            "ub_tp_comm_overlap": config.tp_comm_overlap,
            "ub_bulk_wgrad": config.tp_comm_bulk_wgrad,
            "ub_bulk_dgrad": config.tp_comm_bulk_dgrad,
            "zero_centered_gamma": config.layernorm_zero_centered_gamma,
            "device": 'cpu' if config.use_cpu_initialization else 'cuda',
        }
        te_version = packaging.version.Version(version("transformer-engine"))
        if te_version > packaging.version.Version("1.5.0"):
            # Use old overlap flags if they were supplied instead
            transformer_layer_args["ub_overlap_ag"] = (
                config.tp_comm_overlap_ag
                if hasattr(config, "tp_comm_overlap_ag")
                else config.tp_comm_split_ag or config.tp_comm_atomic_ag
            )
            transformer_layer_args["ub_overlap_rs"] = (
                config.tp_comm_overlap_rs
                if hasattr(config, "tp_comm_overlap_rs")
                else config.tp_comm_split_rs or config.tp_comm_atomic_rs
            )
            if te_version > packaging.version.Version("1.6.0.dev0"):
                transformer_layer_args["ub_overlap_rs_dgrad"] = (
                    config.tp_comm_overlap_rs_dgrad if hasattr(config, "tp_comm_overlap_rs_dgrad") else False
                )
        else:
            transformer_layer_args["ub_split_ag"] = config.tp_comm_split_ag
            transformer_layer_args["ub_split_rs"] = config.tp_comm_split_rs
            transformer_layer_args["ub_atomic_gemm_ag"] = config.tp_comm_atomic_ag
            transformer_layer_args["ub_atomic_gemm_rs"] = config.tp_comm_atomic_rs
        super().__init__(**transformer_layer_args)

        if self.config.enable_cuda_graph and self.training:
            assert not config.cpu_offloading and config.recompute_granularity is None, "Cudagraphs not supported"
            self.add_module('cudagraph_manager', CudaGraphManager())

    # Called by MCore's TransformerBlock.forward
    # megatron/core/transformer/transformer_block.py
    def forward(
        self,
        hidden_states,
        is_first_microbatch=None,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,  # TODO: handle this
    ):
        # Use is_first_microbatch argument during CUDA graph capture. Use self.is_first_microbatch otherwise.
        hidden_states = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            encoder_output=context,
            enc_dec_attn_mask=context_mask,
            inference_params=inference_params,
            is_first_microbatch=is_first_microbatch if is_first_microbatch is not None else self.is_first_microbatch,
            # checkpoint_core_attention,
        )
        self.is_first_microbatch = False
        context = None

        # External CUDA graph requires returned values to be Tensors
        if hasattr(self.config, 'external_cuda_graph') and self.config.external_cuda_graph and self.training:
            return hidden_states
        return hidden_states, context

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = (), metadata=None):
        TENSOR_PARALLEL_LAYERS_AXIS_MAP = {
            'self_attention.layernorm_qkv.weight': 0,
            'self_attention.layernorm_qkv.bias': 0,
            "self_attention.proj.weight": 1,
            "layernorm_mlp.fc1_weight": 0,
            "layernorm_mlp.fc1_bias": 0,
            "layernorm_mlp.fc2_weight": 1,
        }

        state_dict = self.state_dict(prefix='', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            state_dict, prefix, TENSOR_PARALLEL_LAYERS_AXIS_MAP, sharded_offsets
        )

        # TODO: we need to add sharded_state_dict_keys_map to the config. Like in TransformerLayer submodules config
        # prefixed_map = {
        #    f'{prefix}{k}': f'{prefix}{v}'
        #    for k, v in self.config.sharded_state_dict_keys_map.items()
        # }

        # if prefixed_map:
        #    apply_prefix_mapping(sharded_state_dict, prefixed_map)

        return sharded_state_dict

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'cudagraph_manager'):
            return self.cudagraph_manager(self, args, kwargs)
        return super().__call__(*args, **kwargs)


def get_batch_on_this_context_parallel_rank(batch) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    if (cp_size := parallel_state.get_context_parallel_world_size()) > 1:
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
    if (cu_seqlens_argmin := batch.get('cu_seqlens_argmin', None)) is not None:
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


__all__ = [
    "GPTModel",
    "GPTConfig",
    "gpt_data_step",
    "gpt_forward_step",
    "transformer_engine_layer_spec",
    "local_layer_spec",
]
