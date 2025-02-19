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

import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.enums import ModelType

from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.t5.model.t5 import T5Config
from nemo.tron.utils import get_world_size_safe


@dataclass
class RNGConfig:
    seed: int = 1234
    """Random seed used for python, numpy, pytorch, and cuda."""

    te_rng_tracker: bool = False
    """Use the Transformer Engine version of the random number generator.
    Required for CUDA graphs support."""

    inference_rng_tracker: bool = False
    """Use a random number generator configured for inference."""


@dataclass
class RerunStateMachineConfig:
    error_injection_rate: int = 0
    """Rate at which to inject unexpected results, e.g. 1000 means
    once every 1000 result validations"""

    error_injection_type: Literal["correct_result", "transient_error", "persistent_error"] = "transient_error"
    """Type of error to inject. """

    rerun_mode: Literal["disabled", "validate_results", "report_stats"] = "disabled"
    """Use re-run engine to validate results (default) or to emit stats
    on variability of computations due to non-deterministic algorithms."""


@dataclass
class TokenizerConfig:
    vocab_size: Optional[int] = None
    vocab_file: Optional[str] = None
    merge_file: Optional[str] = None
    vocab_extra_ids: int = 0
    tokenizer_type: Optional[
        Literal[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "SentencePieceTokenizer",
            "GPTSentencePieceTokenizer",
            "HuggingFaceTokenizer",
            "Llama2Tokenizer",
            "TikTokenizer",
            "MultimodalTokenizer",
            "NullTokenizer",
        ]
    ] = None
    tokenizer_model: Optional[str] = None
    tiktoken_pattern: Optional[str] = None
    tiktoken_num_special_tokens: int = 1000
    tiktoken_special_tokens: Optional[List[str]] = None
    tokenizer_prompt_format: Optional[str] = None
    special_tokens: Optional[List[str]] = None
    image_tag_type: Optional[str] = None


# TODO (maanug): split this up into modular components


@dataclass
class MegatronLMConfig:
    """MegatronLM config."""

    # ---------------- Network size config. ----------------

    encoder_num_layers: Optional[int] = None
    """Number of encoder transformer layers."""

    decoder_num_layers: Optional[int] = None
    """Number of decoder transformer layers."""

    model_type: ModelType = ModelType.encoder_or_decoder
    """Model architecture type."""

    group_query_attention: bool = False
    """Use group-query attention."""

    max_position_embeddings: Optional[int] = None
    """Maximum number of position embeddings to use. This is the size of position embedding."""

    position_embedding_type: Literal["learned_absolute", "rope", "none"] = "learned_absolute"
    """Position embedding type."""

    use_rotary_position_embeddings: bool = False
    """Use rotary positional embeddings or not. Deprecated: use --position-embedding-type"""

    rotary_base: int = 10000
    """Base to use for rotary positional embeddings, default 10000"""

    rotary_percent: float = 1.0
    """Percent of rotary dimension to use, default 100%%"""

    rotary_seq_len_interpolation_factor: Optional[int] = None
    """Sequence length interpolation factor for rotary embeddings."""

    use_rope_scaling: bool = False
    """Apply rope scaling as used in llama3.1"""

    add_position_embedding: bool = True
    """Disable position embedding. Deprecated: use --position-embedding-type"""

    make_vocab_size_divisible_by: int = 128
    """Pad the vocab size to be divisible by this value. This is added for computational efficieny reasons."""

    openai_gelu: bool = False
    """Use OpenAI's GeLU implementation. This option should not be used unless for backward compatibility reasons."""

    squared_relu: bool = False
    """Use squared relu activation instead of default GeLU"""

    swiglu: bool = False
    """Use gated linear units and SiLU activation instead of default GeLU."""

    onnx_safe: bool = False
    """Use workarounds for known problems with Torch ONNX exporter"""

    untie_embeddings_and_output_weights: bool = False
    """Untie embeddings and output weights."""

    # ---------------- Training config. ----------------

    micro_batch_size: Optional[int] = None
    """Batch size per model instance (local batch size). Global batch size is local batch size times data parallel size times number of micro batches."""

    batch_size: Optional[int] = None
    """Old batch size parameter, do not use. Use --micro-batch-size instead"""

    global_batch_size: Optional[int] = None
    """Training batch size. If set, it should be a multiple of micro-batch-size times data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size as the global batch size. This choice will result in 1 for number of micro-batches."""

    rampup_batch_size: Optional[List[int]] = None
    """Batch size ramp up with the following values:  --rampup-batch-size <start batch size>                       <batch size incerement>                       <ramp-up samples> For example:   --rampup-batch-size 16 8 300000 \    --global-batch-size 1024will start with global batch size 16 and over  (1024 - 16) / 8 = 126 intervals will increase the batch size linearly to 1024. In each interval we will use approximately 300000 / 126 = 2380 samples."""

    decrease_batch_size_if_needed: bool = False
    """If set, decrease batch size if microbatch_size * dp_size does not divide batch_size. Useful for KSO (Keep Soldiering On) to continue making progress if number of healthy GPUs (and corresponding dp_size) does not support current batch_size. Old batch_size will be restored if training is re-started with dp_size that divides batch_size // microbatch_size."""

    recompute_activations: bool = False
    """recompute activation to allow for training with larger models, sequences, and batch sizes."""

    profile: bool = False
    """Enable nsys profiling. When using this option, nsys options should be specified in commandline. An example nsys commandline is `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop`."""

    profile_step_start: int = 10
    """Global step to start profiling."""

    profile_step_end: int = 12
    """Global step to stop profiling."""

    use_pytorch_profiler: bool = False
    """Use the built-in pytorch profiler. Useful if you wish to view profiles in tensorboard."""

    profile_ranks: List[int] = field(default_factory=lambda: [0])
    """Global ranks to profile."""

    record_memory_history: bool = False
    """Record memory history in last rank."""

    memory_snapshot_path: str = "snapshot.pickle"
    """Specifies where to dump the memory history pickle."""

    tp_comm_overlap: bool = False
    """Enables the  overlap of Tensor parallel communication and GEMM kernels."""

    tp_comm_overlap_cfg: Optional[str] = None
    """Config file when tp_comm_overlap is enabled."""

    tp_comm_overlap_ag: bool = True
    """Disables the All-Gather overlap with GEMM by pipelining the GEMM and All-Gather."""

    tp_comm_overlap_rs: bool = True
    """Disables the Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter."""

    tp_comm_overlap_rs_dgrad: bool = False
    """Enables the Reduce-Scatter overlap with dgrad GEMM."""

    tp_comm_bulk_dgrad: bool = True
    """Disables the All-Gather overlap with bprop activation gradient GEMM."""

    tp_comm_bulk_wgrad: bool = True
    """Disables the Reduce-Scatter overlap with bprop weight gradient GEMM."""

    use_cpu_initialization: bool = False
    """If set, initialize weights on the CPU. This eliminates init differences based on tensor parallelism."""

    empty_unused_memory_level: Literal[0, 1, 2] = 0
    """Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation. 0=off, 1=moderate, 2=aggressive."""

    deterministic_mode: bool = False
    """Choose code that has deterministic execution. This usually means slower execution, but is good for debugging and testing."""

    check_weight_hash_across_dp_replicas_interval: Optional[int] = None
    """Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked."""

    train_sync_interval: Optional[int] = None
    """Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU."""

    checkpoint_activations: bool = False
    """Checkpoint activation to allow for training with larger models, sequences, and batch sizes."""

    train_iters: Optional[int] = None
    """Total number of iterations to train over all training runs. Note that either train-iters or train-samples should be provided."""

    train_samples: Optional[int] = None
    """Total number of samples to train over all training runs. Note that either train-iters or train-samples should be provided."""

    exit_interval: Optional[int] = None
    """Exit the program after the iteration is divisible by this value."""

    exit_duration_in_mins: Optional[int] = None
    """Exit the program after this many minutes."""

    exit_signal_handler: bool = False
    """Dynamically save the checkpoint and shutdown the training if SIGTERM is received"""

    bias_gelu_fusion: bool = True
    """Disable bias and GeLU fusion."""

    bias_swiglu_fusion: bool = True
    """Disable bias and swiglu fusion, the fusion is available only when using megatron-core."""

    cross_entropy_loss_fusion: bool = False
    """Enabled fusion of cross entropy loss calculation."""

    use_flash_attn: bool = False
    """use FlashAttention implementation of attention. https://arxiv.org/abs/2205.14135"""

    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = None
    """Single pass vs multiple pass data loader"""

    async_tensor_model_parallel_allreduce: bool = True
    """DEPRECATED. This flag is ignored."""

    persist_layer_norm: bool = True
    """Disable using persistent fused layer norm kernel. This kernel supports only a set of hidden sizes. Please check persist_ln_hidden_sizes if your hidden size is supported."""

    sequence_parallel: bool = False
    """Enable sequence parallel optimization."""

    gradient_accumulation_fusion: bool = True
    """Disable fusing gradient accumulation to weight gradient computation of linear layers"""

    deprecated_use_mcore_models: bool = False
    """DEPRECATED. Use the implementation from megatron core. Now ignored and mcore models are the default, use --use-legacy-models to not use core models."""

    use_legacy_models: bool = False
    """Use the legacy Megatron models, not Megatron-Core models."""

    manual_gc: bool = False
    """Disable the threshold-based default garbage collector and trigger the garbage collection manually. Manual garbage collection helps to align the timing of the collection across ranks which mitigates the impact of CPU-associated jitters. When the manual gc is enabled, garbage collection is performed only at the start and the end of the validation routine by default."""

    manual_gc_interval: int = 0
    """Training step interval to trigger manual garbage collection. When the value is set to 0, garbage collection is not triggered between training steps."""

    manual_gc_eval: bool = True
    """When using manual garbage collection, disable garbage collection at the start and the end of each evaluation run."""

    tp_comm_split_ag: bool = True
    """Disables the All-Gather overlap with fprop GEMM."""

    tp_comm_split_rs: bool = True
    """Disables the Reduce-Scatter overlap with fprop GEMM."""

    # ---------------- Initialization config. ----------------

    data_parallel_random_init: bool = False
    """Enable random initialization of params across data parallel ranks"""

    init_method_xavier_uniform: bool = False
    """Enable Xavier uniform parameter initialization"""

    # ---------------- Checkpointing config. ----------------

    save: Optional[str] = None
    """Output directory to save checkpoints to."""

    save_interval: Optional[int] = None
    """Number of iterations between persistent checkpoint saves."""

    save_optim: bool = True
    """Do not save current optimizer."""

    save_rng: bool = True
    """Do not save current rng state."""

    load: Optional[str] = None
    """Directory containing a model checkpoint."""

    load_optim: bool = True
    """Do not load optimizer when loading checkpoint."""

    load_rng: bool = True
    """Do not load rng state when loading checkpoint."""

    non_persistent_save_interval: Optional[int] = None
    """Number of iterations between non-persistent saves."""

    non_persistent_ckpt_type: Optional[Literal["global", "local", "in_memory", "None"]] = None
    """Type of non-persistent model checkpoints. "global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed. "local" - [TBD] Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk). "in_memory" - [TBD] A special kind of local checkpoint that avoids serialization. None - No non-persistent checkpointing (default option)."""

    non_persistent_global_ckpt_dir: Optional[str] = None
    """Directory containing global non-persistent model checkpoints."""

    non_persistent_local_ckpt_dir: Optional[str] = None
    """Directory containing local non-persistent model checkpoints."""

    non_persistent_local_ckpt_algo: Literal["fully_parallel", "atomic"] = "fully_parallel"
    """Algorithm for local non-persistent checkpointing."""

    finetune: bool = False
    """Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint."""

    pretrained_checkpoint: Optional[str] = None
    """Directory containing a pretrained model checkpoint for finetuning."""

    ckpt_step: Optional[int] = None
    """Checkpoint step to load model from."""

    perform_initialization: bool = True
    """Do not perform initialization when building model, can reduce startup time when definitely loading from a checkpoint"""

    use_checkpoint_args: bool = False
    """Override any command line arguments with arguments from the checkpoint"""

    exit_on_missing_checkpoint: bool = False
    """If '--load' is set, but checkpoint is not found (e.g., path typo), then exit instead of random initialization."""

    use_dist_ckpt_deprecated: bool = False
    """Deprecated: see --ckpt-format."""

    auto_detect_ckpt_format: bool = False
    """Determine if the checkpoint format is in legacy or distributed format. If False, expects distributed checkpoint iff args.ckpt_format != "torch". Might slow down loading a bit (double rank0 ckpt load)."""

    dist_ckpt_format_deprecated: Optional[str] = None
    """Deprecated: see --ckpt-format."""

    ckpt_format: Literal["torch", "torch_dist", "zarr"] = "torch_dist"
    """Checkpoint format to use."""

    ckpt_convert_format: Optional[Literal["torch", "torch_dist", "zarr"]] = None
    """Checkpoint format for conversion."""

    ckpt_convert_save: Optional[str] = None
    """Save directory for converted checkpoint."""

    ckpt_convert_update_legacy_dist_opt_format: bool = False
    """When loading a checkpoint, update the legacy format for the distributed optimizer, which previously used a merged param/grad buffer and a different bucket mapping. The legacy format was deprecated on Feb 13, 2024."""

    ckpt_fully_parallel_save_deprecated: bool = False
    """Deprecated: see --no-ckpt-fully-parallel-save."""

    ckpt_fully_parallel_save: bool = True
    """Disable applying full save parallelization across DP for distributed checkpoints. Depending on ckpt format might decrease the number of files in the checkpoint. Makes DistributedOptimizer checkpoint non-reshardable."""

    async_save: bool = False
    """Apply async checkpointing save. Currently works only with `torch_dist` distributed checkpoint format."""

    ckpt_fully_parallel_load: bool = False
    """Apply full load parallelization across DP for distributed checkpoints."""

    ckpt_assume_constant_structure: bool = False
    """If the model and optimizer state dict structure is constant throughout a *single training job*, it allows for different checkpointing performance optimizations."""

    dist_ckpt_strictness: Literal[
        "assume_ok_unexpected",
        "log_unexpected",
        "log_all",
        "raise_unexpected",
        "raise_all",
        "return_unexpected",
        "return_all",
        "ignore_all",
    ] = "assume_ok_unexpected"
    """Determine handling of key mismatch during checkpoint load. Check StrictHandling docs for flags meaning. NOTE: This flag controls only distributed checkpoint load from storage, not loading state dict into the model."""

    # ---------------- Mixed precision config. ----------------

    fp16_lm_cross_entropy: bool = False
    """Move the cross entropy unreduced loss calculation for lm head to fp16."""

    # ---------------- Distributed config. ----------------

    tensor_model_parallel_size: int = 1
    """Degree of tensor model parallelism."""

    encoder_tensor_model_parallel_size: int = 0
    """Degree of tensor model parallelism for the encoder."""

    pipeline_model_parallel_size: int = 1
    """Degree of pipeline model parallelism."""

    encoder_pipeline_model_parallel_size: int = 0
    """Degree of pipeline model parallelism in the encoder. This is independent of the amount of pipeline in the decoder."""

    pipeline_model_parallel_split_rank: Optional[int] = None
    """Rank where encoder and decoder should be split. Deprecated; use --encoder-pipeline-model-parallel-size instead."""

    model_parallel_size: Optional[int] = None
    """Old model parallel argument, do not use. Use --tensor-model-parallel-size instead."""

    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    """Number of layers per virtual pipeline stage"""

    microbatch_group_size_per_vp_stage: Optional[int] = None
    """Number of contiguous microbatches per virtual pipeline stage"""

    overlap_p2p_comm: bool = True
    """overlap pipeline parallel communication with forward and backward chunks in 1F1B"""

    overlap_p2p_comm_warmup_flush: bool = False
    """if set, overlap pipeline parallel communication in warmup and flush"""

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    """Which backend to use for distributed training."""

    distributed_timeout_minutes: int = 10
    """Timeout minutes for torch.distributed."""

    defer_embedding_wgrad_compute: bool = False
    """If set, defers the vocabulary projection linear layer weight gradient compute to pipeline flush."""

    wgrad_deferral_limit: int = 0
    """Number of micro-batches for which weight gradient computation of vocabulary projection is deferred, defaults to 0 which means all the micro-batches are deferred. Invalid if `defer-embedding-wgrad-compute` is not set"""

    align_grad_reduce: bool = True
    """If not set, all PP stages will launch gradient reduces simultaneously. Otherwise, each PP stage will independently launch as needed."""

    ddp_bucket_size: Optional[int] = None
    """Bucket size for data-parallel communication"""

    ddp_average_in_collective: bool = False
    """If set, average directly in data-parallel communication collective."""

    overlap_param_gather: bool = False
    """If set, overlap param all-gather in distributed optimizer."""

    align_param_gather: bool = True
    """If not set, all PP stages will launch param all-gathers simultaneously. Otherwise, each PP stage will independently launch as needed."""

    scatter_gather_tensors_in_pipeline: bool = True
    """If not set, use scatter/gather to optimize communication of tensors in pipeline."""

    use_ring_exchange_p2p: bool = False
    """If set, use custom-built ring exchange for p2p communications. Note that this option will require a custom built image that support ring-exchange p2p."""

    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    """local rank passed from distributed launcher."""

    lazy_mpu_init: bool = False
    """If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead.Also turns on --use-cpu-initialization flag. This is for external DDP manager."""

    standalone_embedding_stage: bool = False
    """If set, *input* embedding layer is placed on its own pipeline stage, without any transformer layers. (For T5, this flag currently only affects the encoder embedding.)"""

    # num_distributed_optimizer_instances: int = 1
    # """Number of Distributed Optimizer copies across Data Parallel domain."""

    use_torch_fsdp2: bool = False
    """Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel.It is still not in a stable release stage, and may therefore contain bugs or other potential issues."""

    context_parallel_size: int = 1
    """Degree of context parallelism."""

    hierarchical_context_parallel_sizes: Optional[List[int]] = None
    """Degrees of the hierarchical context parallelism. Users should provide a list to specify the sizes for different levels. --hierarchical-context-parallel-sizes 2 4 indicates every two adjacent gpus forms the first level of cp groups and the cp ranks with the same odevity forms the second level of cp groups."""

    nccl_communicator_config_path: Optional[str] = None
    """Path to the yaml file with NCCL communicator configurations. The number of min/max thread groups and thread group cluster size of each communicator can be configured by setting `min_ctas`, `max_ctas`, and `cga_cluster_size`."""

    use_tp_pp_dp_mapping: bool = False
    """If set, distributed ranks initialize order is changed from tp-dp-pp to tp-pp-dp. Make sure EP and CP aren't used with this option enabled"""

    # ---------------- Validation config. ----------------

    eval_iters: int = 100
    """Number of iterations to run for evaluation validation/test for."""

    eval_interval: int = 1000
    """Interval between running evaluation on validation set."""

    skip_train: bool = False
    """If set, bypass the training loop, optionally do evaluation for validation/test, and exit."""

    # ---------------- Data and dataloader config. ----------------

    data_path: Optional[List[str]] = None
    """The weight and prefix list for a set of train, validation, and test datasets which split according to --split. The accepted formats are: (1) a single prefix, (2) a list of weight prefix pairs e.g. weight1 prefix1 weight2 prefix2, (3) a list of prefixes e.g. prefix1 prefix2. For (3), weights are inferred from the lengths of the contributing datasets. This argument is exclusive to the other independent --*-data-path arguments."""

    train_data_path: Optional[List[str]] = None
    """The weight and prefix list for an independent train dataset. Follows the same pattern rules as --data-path."""

    valid_data_path: Optional[List[str]] = None
    """The weight and prefix list for an independent validation dataset. Follows the same pattern rules as --data-path."""

    test_data_path: Optional[List[str]] = None
    """The weight and prefix list for an independent test dataset. Follows the same pattern rules as --data-path."""

    data_cache_path: Optional[str] = None
    """Path to a directory to hold cached index files."""

    mock_data: bool = False
    """Skip data loading and validation and opt for artificial generation of mock data when an implementation is available."""

    retriever_sequence_length: int = 256
    """Maximum sequence length for the biencoder model for retriever"""

    sample_rate: float = 1.0
    """sample rate for training data. Supposed to be 0  < sample_rate < 1"""

    num_workers: int = 8
    """Dataloader number of workers."""

    create_attention_mask_in_dataloader: bool = True
    """If set, do not create attention_masks in dataloader."""

    # ---------------- Tokenizer config. ----------------

    vocab_size: Optional[int] = None
    """Size of vocab before EOD or padding."""

    vocab_file: Optional[str] = None
    """Path to the vocab file."""

    merge_file: Optional[str] = None
    """Path to the BPE merge file."""

    vocab_extra_ids: int = 0
    """Number of additional vocabulary tokens. They are used for span masking in the T5 model"""

    tokenizer_type: Optional[
        Literal[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "SentencePieceTokenizer",
            "GPTSentencePieceTokenizer",
            "HuggingFaceTokenizer",
            "Llama2Tokenizer",
            "TikTokenizer",
            "MultimodalTokenizer",
            "NullTokenizer",
        ]
    ] = None
    """What type of tokenizer to use."""

    tokenizer_model: Optional[str] = None
    """Sentencepiece tokenizer model."""

    tiktoken_pattern: Optional[str] = None
    """Which tiktoken pattern to use. Options: [v1, v2]"""

    tiktoken_num_special_tokens: int = 1000
    """Number of special tokens in tiktoken tokenizer"""

    tiktoken_special_tokens: Optional[List[str]] = None
    """List of tiktoken special tokens, needs to have ["<unk>", "<s>", "</s>"]"""

    # ---------------- Autoresume config. ----------------

    adlr_autoresume: bool = False
    """Enable autoresume on adlr cluster."""

    adlr_autoresume_interval: int = 1000
    """Intervals over which check for autoresume termination signal"""

    # ---------------- Biencoder config. ----------------

    ict_head_size: Optional[int] = None
    """Size of block embeddings to be used in ICT and REALM (paper default: 128)"""

    biencoder_projection_dim: int = 0
    """Size of projection head used in biencoder (paper default: 128)"""

    biencoder_shared_query_context_model: bool = False
    """Whether to share the parameters of the query and context models or not"""

    ict_load: Optional[str] = None
    """Directory containing an ICTBertModel checkpoint"""

    bert_load: Optional[str] = None
    """Directory containing an BertModel checkpoint (needed to start ICT and REALM)"""

    titles_data_path: Optional[str] = None
    """Path to titles dataset used for ICT"""

    query_in_block_prob: float = 0.1
    """Probability of keeping query in block for ICT dataset"""

    use_one_sent_docs: bool = False
    """Whether to use one sentence documents in ICT"""

    evidence_data_path: Optional[str] = None
    """Path to Wikipedia Evidence frm DPR paper"""

    retriever_report_topk_accuracies: List[int] = field(default_factory=lambda: [])
    """Which top-k accuracies to report (e.g. '1 5 20')"""

    retriever_score_scaling: bool = False
    """Whether to scale retriever scores by inverse square root of hidden size"""

    block_data_path: Optional[str] = None
    """Where to save/load BlockData to/from"""

    embedding_path: Optional[str] = None
    """Where to save/load Open-Retrieval Embedding data to/from"""

    indexer_batch_size: int = 128
    """How large of batches to use when doing indexing jobs"""

    indexer_log_interval: int = 1000
    """After how many batches should the indexer report progress"""

    # ---------------- Vision config. ----------------

    num_classes: int = 1000
    """num of classes in vision classificaiton task"""

    num_channels: int = 3
    """Number of channels in input image data"""

    patch_dim: int = 16
    """patch dimension"""

    classes_fraction: float = 1.0
    """training with fraction of classes."""

    data_per_class_fraction: float = 1.0
    """training with fraction of data per class."""

    data_sharding: bool = True
    """Disable data sharding."""

    head_lr_mult: float = 1.0
    """learning rate multiplier for head during finetuning"""

    vision_pretraining: bool = False
    """flag to indicate vision pretraining"""

    vision_pretraining_type: Literal["classify", "inpaint", "dino"] = "classify"
    """pretraining objectives"""

    vision_backbone_type: Literal["vit", "mit", "swin"] = "vit"
    """backbone types types"""

    swin_backbone_type: Literal["tiny", "base", "h3"] = "tiny"
    """pretraining objectives"""

    mask_type: Literal["random", "row"] = "random"
    """mask types"""

    mask_factor: float = 1.0
    """mask size scaling parameter"""

    iter_per_epoch: int = 1250
    """iterations per epoch"""

    dino_local_img_size: int = 96
    """Image size for vision classification task"""

    dino_local_crops_number: int = 10
    """Number of local crops"""

    dino_head_hidden_size: int = 2048
    """Hidden dimension size in dino head"""

    dino_bottleneck_size: int = 256
    """Bottle neck dimension in dino head """

    dino_freeze_last_layer: float = 1
    """Freezing last layer weights"""

    dino_norm_last_layer: bool = False
    """Disable Norm in last layer."""

    dino_warmup_teacher_temp: float = 0.04
    """warump teacher temperature"""

    dino_teacher_temp: float = 0.07
    """teacher temperature"""

    dino_warmup_teacher_temp_epochs: int = 30
    """warmup teacher temperaure epochs"""

    # ---------------- Moe config. ----------------

    expert_model_parallel_size: int = 1
    """Degree of expert model parallelism."""

    moe_extended_tp: bool = False
    """Alternative to expert parallelism, all experts are sharded across TPXEP domain."""

    moe_use_upcycling: bool = False
    """Load a checkpoint of a dense model, convert it into an MoE model, and save the converted model to the path specified by --save. Upcycling is implemented on the top of distributed checkpointing, so it supports parallel modes different from the dense model."""

    # ---------------- Mla config. ----------------

    q_lora_rank: Optional[int] = None
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 32
    """Rank of Key and Value tensors' low rank representation."""

    qk_head_dim: int = 128
    """Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim"""

    qk_pos_emb_head_dim: int = 64
    """Dimension of the position embedding in the QK projection."""

    v_head_dim: int = 128
    """Dimension of the head in the V projection."""

    rotary_scaling_factor: float = 1.0
    """Rotary scaling factor for the rotary embeddings."""

    # ---------------- Straggler config. ----------------

    log_straggler: bool = False
    """If set, tracks and logs straggler per GPU."""

    enable_straggler_on_startup: bool = True
    """If set, StragglerDetector is disabled on startup."""

    straggler_ctrlr_port: int = 65535
    """Port number to toggle StragglerDetector on/off at runtime"""

    straggler_minmax_count: int = 1
    """Number of ranks to report with high/low estimated throughput"""

    # ---------------- Inference config. ----------------

    max_tokens_to_oom: int = 12000
    """Maximum number of tokens during inference tokens here is # in prompt + # to generate Allows us to throw an error before OOM crashes server"""

    output_bert_embeddings: bool = False
    """Output Bert embeddings (via mean pooling) from model, rather than its binary head output or entire hidden batch."""

    bert_embedder_type: Literal["megatron", "huggingface"] = "megatron"
    """Select either Megatron or Huggingface as the Bert embedder."""

    # ---------------- Transformer-engine config. ----------------

    transformer_impl: Literal["local", "transformer_engine"] = "transformer_engine"
    """Which Transformer implementation to use."""

    fp8_param_gather: bool = False
    """Keep the compute param in fp8 (do not use any other intermediate dtype) and perform the param all-gather in fp8."""

    # ---------------- Retro config. ----------------

    retro_project_dir: Optional[str] = None
    """Retro project directory, which contains the preprocessed data for pretraining. This directory is built during preprocessing (see tools/retro/README.md), and contains subdirectories for the chunk database and pretraining neighbors."""

    retro_add_retriever: bool = False
    """Add a retriever to the transformer, for use in pretraining a Retro model."""

    retro_cyclic_train_iters: Optional[int] = None
    """Set number of training iterations for cyclic Retro training."""

    retro_encoder_layers: int = 2
    """Number of layers to use for the retrieval encoder."""

    retro_encoder_hidden_dropout: float = 0.1
    """Hidden dropout for retrieval encoder."""

    retro_encoder_attention_dropout: float = 0.1
    """Attention dropout for retrieval encoder."""

    retro_num_neighbors: int = 2
    """Number of neighbors to retrieve during pretraining."""

    retro_num_retrieved_chunks: int = 2
    """Number of chunks to retrieve from the retrieval database."""

    retro_attention_gate: float = 1
    """Gated cross attention."""

    retro_verify_neighbor_count: bool = True
    """Skip verifying that len(GPT dataset) == len(saved neighbors)."""

    # ---------------- Experimental config. ----------------

    spec: Optional[List[str]] = None
    """Specify the <module_location function_name> pair that returns a spec to customize a model, transformer block, or transformer layer, depending on the use case. To use local spec specify local as the argument. For more details, see the model class, `transformer_block.py`, or `transformer_layer.py`"""

    hybrid_attention_ratio: float = 0.0
    """Ratio of attention layers to total layers, in the range [0.0, 1.0]."""

    hybrid_mlp_ratio: float = 0.0
    """Ratio of mlp layers to total layers, in the range [0.0, 1.0]."""

    hybrid_override_pattern: Optional[str] = None
    """Force a specific hybrid layer pattern. The value should be a string of characters chosen from core.ssm.mamba_hybrid_layer_allocation.Symbols. If a value greater than 0.0 is supplied to any of the hybrid ratio arguments, then the number of each type of layer in the override pattern must match number in the overidden pattern"""

    yaml_cfg: Optional[str] = None
    """Config file to add additional arguments"""

    # ---------------- Ft_package config. ----------------

    enable_ft_package: bool = False
    """If set, Fault Tolerance package is enabled. Note: This feature is for Nvidia internal use only."""

    # ---------------- Config logger config. ----------------


@dataclass(kw_only=True)
class LoggerConfig:
    # ---------------- Logging config. ----------------

    log_interval: int = 100
    """Report loss and timing interval."""

    log_params_norm: bool = False
    """If set, calculate and log parameters norm."""

    log_throughput: bool = False
    """If set, calculate and log throughput per GPU."""

    log_progress: bool = False
    """If set, log progress (in terms of number of processed tokens and number of floating-point operations) to progress.txt file in checkpoint directory."""

    timing_log_level: Literal[0, 1, 2] = 0
    """Granularity level to measure and report timing.    0: report only iteration time and make sure timing       does not introduce extra overhead.   1: report timing for operations that are executed       very limited times (basically once) during       each iteration (such as gradient all-reduce)    2: report timing for operations that migh be       executed numerous times during each iteration. Note that setting the level to 1 or 2 might cause increase in iteration time."""

    timing_log_option: Literal["max", "minmax", "all"] = "minmax"
    """Options for logging timing:  max: report the max timing across all ranks  minmax: report min and max timings across all ranks  all: report timings of all ranks."""

    tensorboard_dir: Optional[str] = None
    """Write TensorBoard logs to this directory."""

    tensorboard_log_interval: int = 1
    """Report to tensorboard interval."""

    tensorboard_queue_size: int = 1000
    """Size of the tensorboard queue for pending events and summaries before one of the 'add' calls forces a flush to disk."""

    log_timers_to_tensorboard: bool = False
    """If set, write timers to tensorboard."""

    log_loss_scale_to_tensorboard: bool = True
    """Disable loss-scale logging to tensorboard."""

    log_validation_ppl_to_tensorboard: bool = False
    """If set, write validation perplexity to tensorboard."""

    log_memory_to_tensorboard: bool = False
    """Enable memory logging to tensorboard."""

    log_world_size_to_tensorboard: bool = False
    """Enable world size logging to tensorboard."""

    wandb_project: Optional[str] = None
    """The wandb project name. Ignore wandb by default."""

    wandb_exp_name: Optional[str] = None
    """The wandb experiment name."""

    wandb_save_dir: Optional[str] = None
    """Path to save the wandb results locally."""

    wandb_entity: Optional[str] = None
    """The wandb entity name."""

    logging_level: Optional[int] = None
    """Set default logging level"""


@dataclass(kw_only=True)
class SchedulerConfig:
    # ---------------- Learning rate config. ----------------
    lr_warmup_steps: int
    lr_decay_steps: int

    lr_decay_style: Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"] = "linear"
    """Learning rate decay function."""

    lr_wsd_decay_style: Literal["exponential", "linear", "cosine"] = "exponential"
    """Decay style for the annealing phase of WSD"""

    lr_decay_iters: Optional[int] = None
    """number of iterations to decay learning rate over, If None defaults to `--train-iters`"""

    lr_decay_samples: Optional[int] = None
    """number of samples to decay learning rate over, If None defaults to `--train-samples`"""

    lr_wsd_decay_samples: Optional[int] = None
    """number of samples for the annealing phase in the wsd schedule"""

    lr_wsd_decay_iters: Optional[int] = None
    """number of iterations for the annealing phase in the wsd schedule"""

    lr_warmup_fraction: Optional[float] = None
    """fraction of lr-warmup-(iters/samples) to use for warmup (as a float)"""

    lr_warmup_iters: int = 0
    """number of iterations to linearly warmup learning rate over."""

    lr_warmup_samples: int = 0
    """number of samples to linearly warmup learning rate over."""

    lr_warmup_init: float = 0.0
    """Initial value for learning rate warmup. The scheduler starts warmup from this value."""

    override_opt_param_scheduler: bool = False
    """Reset the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset."""

    use_checkpoint_opt_param_scheduler: bool = False
    """Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments."""

    # ---------------- Regularization config. ----------------

    start_weight_decay: Optional[float] = None
    """Initial weight decay coefficient for L2 regularization."""

    end_weight_decay: Optional[float] = None
    """End of run weight decay coefficient for L2 regularization."""

    weight_decay_incr_style: Literal["constant", "linear", "cosine"] = "constant"
    """Weight decay increment function."""

    wd_incr_steps: Optional[int] = None
    wsd_decay_steps: Optional[int] = None

    def __post_init__(self):
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        assert self.start_weight_decay >= 0.0
        assert self.end_weight_decay >= self.start_weight_decay

        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, "both override and use-checkpoint are set."


# ---------------- Container config (standalone top-level config) ----------------
@dataclass(kw_only=True)
class ConfigContainer:
    rng_config: RNGConfig = field(default_factory=RNGConfig)
    rerun_state_machine_config: RerunStateMachineConfig = field(default_factory=RerunStateMachineConfig)
    megatron_lm_config: MegatronLMConfig
    model_config: GPTConfig | T5Config
    optimizer_config: OptimizerConfig
    ddp_config: DistributedDataParallelConfig
    scheduler_config: SchedulerConfig
    dataset_config: GPTDatasetConfig
    logger_config: LoggerConfig

    def __post_init__(self):
        # Run validations

        # Distributed
        world_size = get_world_size_safe()
        mlc = self.megatron_lm_config
        encoder_model_size = (
            mlc.encoder_tensor_model_parallel_size
            * mlc.encoder_pipeline_model_parallel_size
            * mlc.context_parallel_size
        )
        decoder_model_size = (
            mlc.tensor_model_parallel_size * mlc.pipeline_model_parallel_size * mlc.context_parallel_size
        )
        total_model_size = encoder_model_size + decoder_model_size
        assert (
            world_size % total_model_size == 0
        ), f"world size ({world_size}) is not divisible by total_model_size ({encoder_model_size=} + {decoder_model_size=})"
        self.data_parallel_size = world_size // total_model_size

        self.megatron_lm_config.use_cpu_initialization = (
            self.megatron_lm_config.use_cpu_initialization or self.megatron_lm_config.lazy_mpu_init
        )

        # Scheduler
        if self.scheduler_config.lr_decay_iters is None:
            self.scheduler_config.lr_decay_iters = self.megatron_lm_config.train_iters
        self.scheduler_config.lr_decay_steps = (
            self.scheduler_config.lr_decay_iters * self.megatron_lm_config.global_batch_size
        )
        self.scheduler_config.wd_incr_steps = (
            self.megatron_lm_config.train_iters * self.megatron_lm_config.global_batch_size
        )
        self.scheduler_config.wsd_decay_steps = None
        if self.scheduler_config.lr_wsd_decay_iters is not None:
            self.scheduler_config.wsd_decay_steps = (
                self.scheduler_config.lr_wsd_decay_iters * self.megatron_lm_config.global_batch_size
            )
        if self.scheduler_config.lr_warmup_fraction is not None:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_fraction * self.scheduler_config.lr_decay_steps
            )
        else:
            self.scheduler_config.lr_warmup_steps = (
                self.scheduler_config.lr_warmup_iters * self.megatron_lm_config.global_batch_size
            )
