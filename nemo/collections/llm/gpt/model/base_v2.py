from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, List, Tuple

import functools
import itertools
import os
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Literal, Mapping, Optional, Sized, Union
from omegaconf import OmegaConf

import pytorch_lightning as L
from pytorch_lightning import Trainer
import torch
import torch.distributed
from torch.optim import Optimizer

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal, register_function, get_function_from_registry

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_outplace
from megatron.core.dist_checkpointing.mapping import LocalNonpersitentObject
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from megatron.core.dist_checkpointing.strategies import tensorstore
from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from megatron.core.transformer.transformer_layer import TransformerLayer as MCoreTransformerLayer
from nemo.utils.callbacks.dist_ckpt_io import DistributedCheckpointIO



from nemo.lightning import get_vocab_size, io
# from nemo.lightning.base import ModelConfig
from nemo.lightning.megatron_parallel import MaskedTokenLossReduction

from nemo.core import ModelPT
from nemo.collections.nlp.models.nlp_model import NLPModel, NLPSaveRestoreConnector
from nemo.core.config.modelPT import OptimConfig, SchedConfig

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.transformer import AutocastTransformerLayer, ParallelTransformerLayer
from nemo.collections.nlp.parts import utils_funcs
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.optim import MainParamsOptimizerWrapper
from nemo.core.optim.optimizers import init_optimizer_states
from nemo.utils import AppState, logging
from nemo.utils.model_utils import ckpt_to_dir, inject_model_parallel_rank, uninject_model_parallel_rank

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel

    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


    
@dataclass
class GPTOptimConfig(OptimConfig):
    name: str = "fused_adam"
    lr: float = 1e-4
    weight_decay: float = 0.0

    sched: Optional[SchedConfig] = None


# @dataclass
# class GPTConfigV2(TransformerConfig, ModelConfig):  # 
    # From megatron.core.models.gpt.gpt_model.GPTModel
    # fp16_lm_cross_entropy: bool = False
    # parallel_output: bool = True
    # share_embeddings_and_output_weights: bool = False
    # make_vocab_size_divisible_by: int = 128
    # position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute"
    # rotary_base: int = 10000
    # rotary_percent: float = 1.0
    # seq_len_interpolation_factor: Optional[float] = None
    # seq_length: int = 1024

    # # TODO: Move this to better places?
    # get_attention_mask_from_fusion: bool = False

    # window_size: Any = None
    # optimizer_fn: Optional[Callable[["GPTModel"], Optimizer]] = None

#     init_method: Optional[Callable[["str"], None]] = None
#     output_layer_init_method: Optional[Callable[["str"], None]] = None
#     activation_func: Optional[Callable[["torch.Tensor"], "torch.Tensor"]] = None

    # def configure_model(self, tokenizer) -> "MCoreGPTModel":
    #     vp_size = self.virtual_pipeline_model_parallel_size
    #     if vp_size:
    #         p_size = self.pipeline_model_parallel_size
    #         assert (
    #             self.num_layers // p_size
    #         ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

    #     from megatron.core import parallel_state
    #     from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    #     from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
        
    #     return MCoreGPTModel(
    #         self,
    #         transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
    #         vocab_size=get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by),
    #         max_sequence_length=self.seq_length,
    #         fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
    #         parallel_output=self.parallel_output,
    #         share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
    #         position_embedding_type=self.position_embedding_type,
    #         rotary_percent=self.rotary_percent,
    #         rotary_base=self.rotary_base,
    #         seq_len_interpolation_factor=self.seq_len_interpolation_factor,
    #         pre_process=parallel_state.is_pipeline_first_stage(),
    #         post_process=parallel_state.is_pipeline_last_stage(),
    #     )
    

#     optim: GPTOptimConfig = GPTOptimConfig()

#     def __iter__(self):
#         keys = set([f.name for f in fields(self)])
#         yield keys


def dataclass_from_dict(klass, d):
    try:
        print(klass)
        fieldtypes = {f.name:f.type for f in fields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except Exception as e:
        return d  # Not a dataclass field


@dataclass
class ModelParallelConfigNeMo:
    """Base configuration for Megatron Core

    The initialization function has an argument for each parameter.
    """

    ###################
    # Model parallelism
    ###################
    tensor_model_parallel_size: int = 1
    """Intra-layer model parallelism. Splits tensors across GPU ranks."""

    pipeline_model_parallel_size: int = 1
    """Inter-layer model parallelism. Splits transformer layers across GPU ranks."""

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """Interleaved pipeline parallelism is used to improve performance by reducing the pipeline
       bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
       The number of virtual blocks per pipeline model parallel rank is the virtual model parallel
       size.  See Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM:
       arxiv.org/pdf/2104.04473.pdf for more details.
    """

    sequence_parallel: bool = False
    """Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer norms
       and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer Models
       (https://arxiv.org/abs/2205.05198) for more details.
    """

    context_parallel_size: int = 1
    """Splits network input along sequence dimension across GPU ranks."""

    expert_model_parallel_size: int = 1
    """Distributes Moe Experts across sub data parallel dimension."""

    moe_extended_tp: bool = False
    """Alternative parallelization strategy for expert parallelism. Instead of distributing experts
       across expert_model_parallel_size, each expert is sharded along extendended tensor parallel
       domain (tensor_model_paralle_size * expert_model_parallel_size). It avoids the load balancing
       problem with MOE training. 
    """

    ###################
    # Initialization
    ###################
    perform_initialization: bool = True
    """If true, weights are initialized. This option can be useful when you know you are going to
       load values from a checkpoint.
    """

    use_cpu_initialization: bool = False
    """When set to False, we initialize the weights directly on the GPU. CPU initialization is the
       same regardless of tensor model parallelism, but GPU initialization is not. Transferring
       weights from CPU to GPU can take a significant amount of time for large models.
    """

    ###################
    # Training
    ###################
    fp16: bool = False
    """If true, train with fp16 mixed precision training."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training."""

    params_dtype: str = 'float32'
    """dtype used when intializing the weights."""

    timers: Optional[str] = None
    """Timers object to call for various timing functions. See megatron.core.timers.Timers"""

    finalize_model_grads_func: Optional[str] = None
    """Function that finalizes gradients on all workers. Could include ensuring that grads are
       all-reduced across data parallelism, pipeline parallelism, and sequence parallelism
       dimensions.
    """

    grad_scale_func: Optional[str] = None
    """If using loss scaling, this function should take the loss and return the scaled loss. If
       None, no function is called on the loss.
    """

    no_sync_func: Optional[str] = None
    """Function that creates a context that suppresses asynchronous data-parallel communication. If
       the model is an instance of core.distributed.DistributedDataParallel, the default is to use
       core.distributed.DistributedDataParallel.no_sync.
    """

    grad_sync_func: Optional[str] = None
    """Function that launches asynchronous gradient reductions (e.g. distributed optimizer gradient
       reduce-scatters). The function should take one argument: an iterable of parameters whose
       gradients are to be synchronized.
    """

    param_sync_func: Optional[str] = None
    """Function that launches asynchronous parameter synchronizations (e.g. distributed optimizer
       parameter all-gathers). The function should take one argument: an iterable of parameters to
       be synchronized.
    """

    deterministic_mode: bool = False
    """If true, code that has deterministic execution will be chosen. This usually
       means slower execution, but is good for debugging and testing. Defaults to False."""

    enable_autocast: bool = False
    """If true runs the forward step function inside torch.autocast context."""

    autocast_dtype: Optional[str] = None
    """dtype to pass to torch.amp.autocast when enabled. If None, is set to pipeline_dtype."""

    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None
    """If int, set the number of microbatches where not all of the layers will be checkpointed and
       recomputed. The rest of the microbatches within the window of maximum outstanding
       microbatches will recompute all layers (either full recompute or selective recompute). If
       None, the checkpoint and recompute will be left up to the forward_step function.

    """

    ###################
    # Optimizations
    ###################
    gradient_accumulation_fusion: bool = False
    """If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA extension
       fused_weight_gradient_mlp_cuda module. To use gradient_accumulation_fusion you must install
       APEX with --cpp_ext and --cuda_ext. For example: "pip install --global-option=\"--cpp_ext\"
       --global-option=\"--cuda_ext\" ". Note that the extension requires CUDA>=11. Otherwise, you
       must turn off gradient accumulation fusion.
    """

    async_tensor_model_parallel_allreduce: bool = False
    """NOTE: Deprecated. This flag is ignored."""

    use_te_rng_tracker: bool = False
    """If true, uses RNG state tracker in TransformerEngine if exists.
    """

    tp_comm_overlap: bool = False
    """If true, allows overlapping of Linear layer execution with tensor parallel communication
       collectives like AllGather/ReduceScatter. Overlapping is done for the linear layers wherever
       possible during the forward and the backward pass.
    """

    tp_comm_bulk_wgrad: bool = True
    """If true, allows All-Gather overlap with Bprop activation gradient GEMM. Don't care if
       tp_comm_overlap is False.
    """

    tp_comm_bulk_dgrad: bool = True
    """If true, allows Reduce-Scatter overlap with Bprop weight gradient GEMM. Don't care if
       tp_comm_overlap is False.
    """

    tp_comm_overlap_ag: bool = True
    """If true, allows All-Gather overlap with GEMM by pipelining the GEMM and All-Gather.
       Don't care if tp_comm_overlap is False.
    """

    tp_comm_overlap_rs: bool = True
    """If true, allows Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter.
       Don't care if tp_comm_overlap is False.
    """

    tp_comm_overlap_rs_dgrad: bool = False
    """If true, allows Reduce-Scatter overlap with DGRAD GEMM by pipelining the
       GEMM and Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
    """

    tp_comm_split_ag: bool = True
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather
       splits. Don't care if tp_comm_overlap is False.
    """

    tp_comm_atomic_ag: bool = False
    """Deprecated from TransformerEngine v1.6.0.
        If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather both
       done atomically. Don't care if tp_comm_overlap is False.
    """

    tp_comm_split_rs: bool = True
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
       Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
    """

    tp_comm_atomic_rs: bool = False
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
       Reduce-Scatter both done atomically. Don't care if tp_comm_overlap is False.
    """

    ###################
    # Pipeline Parallel
    ###################
    pipeline_dtype: Optional[str] = None
    """dtype used in p2p communication, usually params_dtype"""

    variable_seq_lengths: bool = False
    """Support for variable sequence lengths across microbatches. Setting this communicates the size
        of tensors during pipeline parallelism communication, because of this extra overhead it
        should only be set if the sequence length varies by microbatch within a global batch.
    """

    overlap_p2p_comm: bool = False
    """When True some of the peer to peer communication for pipeline parallelism will overlap with
       computation. Must be False if batch_p2p_comm is true.
    """

    batch_p2p_comm: bool = True
    """Use batch_isend_irecv instead of individual isend/irecv calls. Must be False if
       overlap_p2p_comm is True.
    """

    batch_p2p_sync: bool = True
    """When using batch_isend_irecv, do a cuda.device.synchronize afterward to work around a bug in
       older version of PyTorch.
    """

    use_ring_exchange_p2p: bool = False
    """Use custom ring_exchange kernel instead of torch.distributed.batch_isend_irecv(). Requires
       custom built torch with torch.distributed.ring_exchange.
    """

    deallocate_pipeline_outputs: bool = False
    """If True, output data is deallocated after the tensor is sent to the next pipeline stage.
       Helps with saving memory, does nothing when pipeline parallel is not used.
    """

    defer_embedding_wgrad_compute: bool = False
    """If true, defers the embedding WGRAD GEMMs while pipeline flush is
       taking place enabling us to hide pipeline flush latency. Defaults to False.
    """

    pipeline_model_parallel_split_rank: Optional[int] = None
    """If int, rank where encoder and decoder should be split in cases where the model has both an
       encoder and decoder (e.g., T5). Ignored if None.
    """

    ###################
    # CPU Offloading
    ###################
    cpu_offloading: bool = False
    """When set to True, all the activations are offloaded to the CPU asynchronously."""

    cpu_offloading_num_layers: int = 0
    """Tells the number of transformer layers for which activations has to be offloaded."""

    _cpu_offloading_context: Any = None  # Used for internal use only, not to be set by the user. TODO: Need to move to the 'right' place when possible.
    """For internal use only, do not set."""

    cpu_offloading_activations: bool = True
    """If True, offloads the activations to CPU."""

    cpu_offloading_weights: bool = True
    """If True, offloads the weights to CPU."""

    ###################
    # Timing
    ###################
    barrier_with_L1_time: bool = True
    """If true, use barrier with level 1 time measurements. It is up to the user to make sure
       calling barrier with their timers will not result in hangs. This can happen if for example
       the user adds a level 1 timer that is not called by all ranks.
    """

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.sequence_parallel:
            if self.tensor_model_parallel_size <= 1:
                raise ValueError("Can not use sequence paralllelism without tensor parallelism")

        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError(
                    "When using pipeline parallelism, pipeline_dtype must be specified"
                )

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype

        if self.defer_embedding_wgrad_compute and self.pipeline_model_parallel_size == 1:
            raise ValueError(
                "Cannot defer embedding wgrad compute when pipeline model parallel is not used"
            )

        if self.defer_embedding_wgrad_compute and not self.gradient_accumulation_fusion:
            raise ValueError(
                "Cannot defer embedding wgrad compute when gradient accumulation fusion is not used"
            )

        if self.expert_model_parallel_size > 1 and self.tensor_model_parallel_size > 1:
            if self.sequence_parallel is False:
                raise ValueError(
                    "When using expert parallelism and tensor parallelism, sequence parallelism must be used"
                )



@dataclass
class GPTConfigV2(ModelParallelConfigNeMo):
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


    ####################
    # model architecture
    ####################
    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: Optional[int] = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size if not provided."""

    kv_channels: Optional[int] = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residule connection ordering."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    add_bias_linear: bool = True
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    activation_func: str = 'gelu'
    """Activation function to use for the non-linearity in the MLP."""

    activation_func_fp8_input_store: bool = False
    """Store the input of MLP activation function in FP8 for backprop to save memory.
    The stored input is casted back to the original precision before backprop compuatation."""

    num_moe_experts: Optional[int] = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    window_size: Optional[List[int]] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    normalization: str = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    qk_layernorm: bool = False
    """Whether to apply LayerNorm to the query and key embeddings."""

    test_mode: bool = False
    """Whether to run real-time tests."""

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    ####################
    # initialization
    ####################
    init_method: Optional[str] = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: Optional[str] = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel. This kernel only supports a fixed set
    of hidden sizes."""

    memory_efficient_layer_norm: bool = False
    """If True, and using local layers (not from TransformerEngine), tells Apex to use the memory
    efficient fused LayerNorm kernel. Ignored if not using LayerNorm."""

    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: Optional[str] = None
    recompute_granularity: Optional[str] = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where only the memory intensive part of attention is checkpointed.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: Optional[str] = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: Optional[int] = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    distribute_saved_activations: Optional[bool] = None
    """If True, distribute recomputed activations across the model parallel group."""

    ####################
    # fp8 related
    ####################
    fp8: Optional[str] = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """Controls how often the scaling factor is recomputed."""

    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.

    """

    fp8_wgrad: bool = True
    """When set to False, override FP8 config options and do the wgrad computation in higher precision."""

    fp8_dot_product_attention: bool = False
    """When set to True, use the FP8 implementation of Dot Product Attention."""

    fp8_multi_head_attention: bool = False
    """When set to True, use the FP8 implementation of Multi Head Attention."""

    ####################
    # MoE related
    ####################
    moe_router_load_balancing_type: str = "aux_loss"
    """Determines the load balancing strategy for the router. "aux_loss" corresponds to the load
    balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing
    algorithm used in S-BASE, and "none" implies no load balancing."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).

    """

    moe_aux_loss_coeff: float = 0  # 1e-2 would be a good start value for load balance loss.
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: Optional[float] = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: Optional[float] = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    moe_token_dropping: bool = False  # TODO: Support token dropping.
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather' and 'alltoall'."""
    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: Optional[float] = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False."""

    moe_token_drop_policy: str = 'probs'
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
    """
    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    enable_cuda_graph: bool = False
    """When set to true, TransformerLayer blocks are wrapped with CUDA graph."""


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
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError(f'num_moe_experts must be non None to use expert-parallel.')

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError(f'num_moe_experts must be non-negative.')

        if self.moe_expert_capacity_factor is not None:
            if self.moe_token_dispatcher_type != "alltoall":
                raise ValueError(
                    f'moe_expert_capacity_factor only works with alltoall token dispatcher'
                )
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["aux_loss", "none"]:
                raise ValueError(
                    f'moe_expert_capacity_factor only works with aux_loss or none load balancing'
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    f'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
                )

        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f'CPU offloading can be done only for layers less than {self.num_layers}'
            )

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                f'Currently there is no support for Pipeline parallelism with CPU offloading'
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                f'CPU offloading does not work when activation recomputation is enabled'
            )

        if self.recompute_granularity is not None:
            if not self.recompute_granularity in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
                )

            if self.recompute_method is not None:
                if not self.recompute_method in ['block', 'uniform']:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != 'selective':
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be between '
                    f'1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}'
                )
            elif (
                self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be None.'
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel}'
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f'num_layers: {self.num_layers} must be divisible by virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_activation_fusion:
            if self.activation_func not in [F.gelu, F.silu]:
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either gelu or swiglu"
                )
            if (
                self.activation_func == F.gelu
                and not self.gated_linear_unit
                and not self.add_bias_linear
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False, "
                    "and activation function is gelu, add_bias_linear must also be True."
                )
        if self.activation_func_fp8_input_store:
            if self.activation_func != F.silu or not self.gated_linear_unit:
                raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")
        if self.apply_rope_fusion and self.rotary_interleaved:
            raise ValueError(f'rotary_interleaved does not work with apply_rope_fusion.')

        # Register a default function for initialization and restoration
        init_method_fn = init_method_normal(self.init_method_std)
        register_function(init_method_fn)

        if self.init_method is None:
            self.init_method =  init_method_fn.__name__  # init_method_normal(self.init_method_std)

        # Register a default function for initialization and restoration
        scaled_init_method_normal_fn = scaled_init_method_normal(
            self.init_method_std, self.num_layers
        )
        register_function(scaled_init_method_normal_fn)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal_fn.__name__
            # scaled_init_method_normal(
            #     self.init_method_std, self.num_layers
            # )

        if self.moe_extended_tp:
            if self.moe_token_dispatcher_type != 'allgather':
                raise ValueError(
                    "Moe extended TP parallelism only applies to allgather based token dispatcher."
                )
            extended_tp_size = self.tensor_model_parallel_size * self.expert_model_parallel_size
            if self.ffn_hidden_size % extended_tp_size != 0:
                raise ValueError(
                    f'ffn_hidden_size: {self.ffn_hidden_size} must be divisible by extended_tp_size {extended_tp_size}'
                )

class LLMSaveRestoreConnector(SaveRestoreConnector):

    def __init__(self):
        super().__init__()
        self.pack_nemo_file = False  # Only save unpacked checkpoint

    def save_to(self, model, save_path: str):
        app_state = AppState()

        print("Save path", save_path)
        # # Check if using distributed checkpointing
        # if model.cfg.get("fsdp", False):
        #     dist_ckpt = False
        # else:
        #     dist_ckpt = hasattr(model, 'sharded_state_dict') and model.sharded_state_dict() is not None

        dist_ckpt = True
        dist_ckpt_dir = None

        if '.nemo' in save_path and not self.pack_nemo_file:
            # save_path = save_path[:-5]
            # dir_name = save_path
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


class GPTModelV2(ModelPT,  # NLPModel, # ModelPT, 
                 # io.IOMixin, 
                 # io.ConnectorMixin
                 ):
    def __init__(
        self,
        cfg: GPTConfigV2,
        trainer: L.Trainer = None,
    ):
        # If you ned the dataclass itself for its util methods, we can rebuild it due to proper serialization to yaml
        if not isinstance(cfg, GPTConfigV2):
            model_cfg = OmegaConf.to_object(cfg)
            model_cfg.pop('nemo_version', None)
            model_cfg.pop('target', None)
            cfg = dataclass_from_dict(GPTConfigV2, model_cfg)

        # config
        super().__init__(cfg, trainer=trainer)

        self._save_restore_connector = LLMSaveRestoreConnector()

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
        if not hasattr(self, 'module'):
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
        return self.__class__(self.config, self.trainer, self.tokenizer)
    
    def setup_training_data(self, cfg):
        pass

    def setup_validation_data(self, cfg):
        pass

    def setup_test_data(self, cfg):
        pass

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
