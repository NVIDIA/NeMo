# NeMo Recipe Configuration Map

- TransformerConfig and ModelParallelConfig defintions (used in recipe -> model -> config -> GPTConfig) can be found here-
  -  https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py 
  -  https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/model_parallel_config.py

- The parameter values are defaults as defined in the base class for a module.
- Click on the links to see complete list of configuration options for a module.

<details open>
<summary>recipe</summary>
<blockquote>

<details open>
  <summary>trainer (pytorch_lightning.Trainer)</summary>

  <blockquote>

  <details open><summary>strategy <a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/strategies/megatron_strategy.py">(MegatronStrategy)</a></summary>
  <blockquote>

  ```sh
  tensor_model_parallel_size: int = 1 # Intra-layer model parallelism. Splits tensors across GPU ranks
  pipeline_model_parallel_size: int = 1 # Inter-layer model parallelism. Splits transformer layers across GPU ranks
  virtual_pipeline_model_parallel_size: Optional[int] = None # Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size
  context_parallel_size: int = 1 # Splits network input along sequence dimension across GPU ranks
  sequence_parallel: bool = False # Parallelizes layer norms and dropout sequentially
  expert_model_parallel_size: int = 1 # Distributes Moe Experts across sub data parallel dimension
  pipeline_dtype: Optional[torch.dtype] = None # dtype used in p2p communication
  ```    
  <details><summary>ddp: Union[DDPLiteral, <a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel_config.py">DistributedDataParallelConfig</a>] = "megatron"</summary>
  <blockquote>

  ```sh
  grad_reduce_in_fp32: bool = False # If true, reduce grads in fp32; this is overridden by same config in MegatronMixedPrecision
  overlap_grad_reduce: bool = False # If true, overlap grad all-reduce / reduce-scatter with backward compute
  overlap_param_gather: bool = False # If true, overlap param all-gather with forward compute
  align_param_gather: bool = False # If true, all PP stages will launch param all-gathers simultaneously
  use_distributed_optimizer: bool = False # If true, issue reduce-scatter collectives to aggregate gradients and clean up originally allocated model parameters, otherwise issue all-reduce collectives ; this is overridden by same config in OptimizerConfig
  check_for_nan_in_grad: bool = False # If true, check for NaNs in gradients _before_ communication collective
  bucket_size: Optional[int] = None # Maximum number of parameters in each bucket
  average_in_collective: bool = False # If true, compute average in collective directly, as opposed to dividing by the dp_size first and then computing sum in the collective
  fp8_param_gather: bool = False # If true, keep the compute param in fp8 (do not use any other intermediate dtype) and perform the param all-gather in fp8
  ```  
  </blockquote>
  </details>
  </blockquote>
  </details>

  <details>
  <summary>callbacks (Optional[List[Callback]]=None)</summary>
  <blockquote>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/exp_manager.py#L240">TimingCallback</a> (Callback)</summary>
  <blockquote>

  ```sh
  reduction: str = "mean" # reduction over multiple timings of the same timer
  sync_cuda: bool = False # if True torch.cuda.synchronize() is called for start/stop
  buffer_size: int = 1 # if positive, limits the number of stored measures per name
  ```
  </blockquote>
  </details>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/callbacks/megatron_comm_overlap.py">MegatronCommOverlapCallback</a> (Callback)</summary>
  <blockquote>

  ```sh
  tp_comm_overlap: bool = None # Enable tensor parallel overlap (experimental)
  tp_comm_overlap_cfg: TransformerLayerTPOverlapCfg = None # Tensor parallel overlap config
  overlap_p2p_comm: bool = None # Enable pipeline parallel overlap
  batch_p2p_comm: bool = None # Batch pipeline parallel send/recv into a single op
  overlap_grad_reduce: bool = None # Overlap data parallel gradient reduction with compute
  overlap_param_gather: bool = None # Overlap data parallel parameter gather with compute
  overlap_param_gather_with_optimizer_step: bool = None # Overlap data parallel parameter gather optimizer step
  align_param_gather: bool = None # Align data parallel parameter gather across virtual pipeline chunks
  bucket_size: int = None # The DDP bucket size, controls the data parallel overlap granularity
  defer_embedding_wgrad_compute: bool = None # Overlap wgrads with the pipeline drain bubble for the last pipeline stage
  wgrad_deferral_limit: int = None # Limit of how many outstanding wgrads may be overlapped with the pipeline drain bubble
  ```
  </blockquote>
  </details>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/callbacks/nsys.py">NsysCallback</a> (Callback)</summary>
  <blockquote>

  ```sh
  start_step: int # Global batch to start profiling
  end_step: int # Global batch to end profiling
  ranks: List[int] = [0] # Global rank IDs to profile
  gen_shape: bool = False # Generate model and kernel details including input shapes
  ```
  </blockquote>
  </details>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/callbacks/memory_profiler.py">MemoryProfileCallback</a> (Callback)</summary>
  <blockquote>

  ```sh
  dir: str = "/mem_profile" # Directory to store the memory profile dump
  warn_cycles: bool = True # Whether to enable [reference cycle detection](https://pytorch.org/blog/understanding-gpu-memory-2/)
  ranks: List = [] # List of ranks to collect snapshot on, defaults to all if list is empty
  ```
  </blockquote>
  </details>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/callbacks/garbage_collection.py">GarbageCollectionCallback</a> (Callback)</summary>
  <blockquote>

  ```sh
  gc_interval_train: int # Number of global train steps at which garbage collection is done
  gc_interval_val: int # Number of global validation steps at which garbage collection is done
  ```
  </blockquote>
  </details>
  </blockquote>
  </details>

  <details>
  <summary>plugins (nemo_run.Plugin)</summary>
  <blockquote>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/plugins/mixed_precision.py">MegatronMixedPrecision</a></summary>
  <blockquote>

  ```sh
  precision: Literal["16-mixed", "bf16-mixed", "32"] # dtype for mixed precision training
  params_dtype: torch.dtype = None # dtype used when intializing the weights
  pipeline_dtype: torch.dtype = None # dtype used in p2p communication, usually params_dtype
  autocast_enabled: bool = False
  grad_reduce_in_fp32: bool = True # If true, reduce grads in fp32; this overrides same config in DistributedDataParallelConfig
  fp8: str = None # If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined choices- 'e4m3' and 'hybrid'
  fp8_margin: int = 0 # Margin for the scaling factor computation
  fp8_amax_history_len: int = 1 # The length of the amax history window used for scaling factor computation
  fp8_amax_compute_algo: str = "most_recent" # Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2 predefined choices- `max` and 'most_recent'
  fp8_params: bool = False # fp8 dtype weights, sets 'transformer_engine.pytorch.fp8.FP8GlobalStateManager.FP8_PARAMETERS=True' and 'fp8_param_gather=True'
  ```
  </blockquote>
  </details>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/run/plugins.py">PerfEnvPlugin</a></summary>
  <blockquote>

  ```sh
  enable_layernorm_sm_margin: bool = True # Set SM margin for TransformerEngine's Layernorm, so in order to not block DP level communication overlap
  layernorm_sm_margin: int = 16 # The SM margin for TransformerEngine Layernorm
  enable_vboost: bool = False # Whether to steer more power towards tensor cores via `sudo nvidia-smi boost-slider --vboost 1`. May not work on all systems
  ```
  </blockquote>
  </details>

  </blockquote>
  </details>

  </blockquote>

</details>

<details>
  <summary>model (pytorch_lightning.LightningModule)</summary>

  <blockquote>
    
  <details><summary>config (<a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/model/base.py">GPTConfig</a>)</summary>
  <blockquote>

  ```sh
  seq_length: int = 1024 # Number of tokens in a single sequence
  attention_softmax_in_fp32: bool = False # If True, run attention masking and softmax in fp32. This should be True if apply_query_key_layer_scaling is True
  num_layers: int = 0 # Number of transformer layers in a transformer block
  hidden_size: int = 0 # Transformer hidden size
  num_attention_heads: int = 0 # Number of transformer attention heads
  num_query_groups: Optional[int] = None # Number of query groups for group query attention. If None, normal attention is used
  ffn_hidden_size: Optional[int] = None # Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size if not provided
  hidden_dropout: float = 0.1 # Dropout probability for transformer hidden state
  attention_dropout: float = 0.1 # Post attention dropout probability
  add_bias_linear: bool = True # Include a bias term in all linear layers (QKV projections, after core attention, and two in MLP layer)
  gated_linear_unit: bool = False # Use a gated linear unit for the first linear layer in the MLP
  activation_func: Callable = F.gelu # Activation function to use for the non-linearity in the MLP
  normalization: bool = "LayerNorm" # Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`
  layernorm_epsilon: float = 1e-5 # Epsilon value for any LayerNorm operations
  layernorm_zero_centered_gamma: bool = False # If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves numerical stability

  # Fusions
  masked_softmax_fusion: bool = True # If True, uses softmax fusion
  cross_entropy_loss_fusion: bool = True # If this is enabled, the fused cross entropy implementation would be used
  gradient_accumulation_fusion: bool = _grad_accum_fusion_available # If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA extension fused_weight_gradient_mlp_cuda module
  bias_activation_fusion: bool = False # If True, fuses bias addition and the activation function when possible
  bias_dropout_fusion: bool = False # If True, uses bias dropout fusion
  apply_rope_fusion: bool = False # If True, use fused RoPE kernel

  recompute_granularity: Optional[str] = None # Determines which type of activation recompute to use. If set, must be 'selective' or 'full'.
  recompute_method: Optional[str] = None # Determines which transformer layers will be recomputed. If set, must be 'uniform' or 'block'.
  recompute_num_layers: Optional[int] = None # If True, distribute recomputed activations across the model parallel group
  distribute_saved_activations: Optional[bool] = None # If True, distribute recomputed activations across the - model parallel group.

  enable_cuda_graph: bool = False # When set to true, TransformerLayer layers are swapped with a CUDA graphed version.
  external_cuda_graph: bool = False # When set to true, TransformerLayer layers are swapped with user provided CUDA graphs.
  ```
  </blockquote>
  </details>

  </blockquote>

</details>

<details>
  <summary>data (pytorch_lightning.LightningDataModule)</summary>

  <blockquote>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/data/mock.py">MockDataModule</a></summary>
  <blockquote>
      
  ```sh
  seq_length: int = 2048 # Number of tokens in a single sequence
  tokenizer: Optional["TokenizerSpec"] = None # TokenizerSpec object to convert sequences to tokens
  micro_batch_size: int = 4 # The size of each micro batch
  global_batch_size: int = 8 # The size of each micro batch
  rampup_batch_size: Optional[List[int]] = None # Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples]
  num_train_samples: int = 10_000 # The number of samples to use for training
  num_val_samples: int = 10_000 # The number of samples to use for validation
  num_test_samples: int = 10_000 # The number of samples to use for testing
  num_workers: int = 8 # How many subprocesses to use for data loading
  pin_memory: bool = True # If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them
  persistent_workers: bool = False # If True, the data loader will not shut down the worker processes after a dataset has been consumed once
  create_attention_mask: bool = False # Option to enable the attention masks generation
  ```
  </blockquote>
  </details>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/data/squad.py">SquadDataModule</a> (For fine-tuning jobs)</summary>
  <blockquote>

  ```sh
  seq_length: int = 2048 # Number of tokens in a single sequence
  tokenizer: Optional["TokenizerSpec"] = None # TokenizerSpec object to convert sequences to tokens
  micro_batch_size: int = 4 # The size of each micro batch
  global_batch_size: int = 8 # The size of each micro batch
  ```
  <details><summary>packed_sequence_specs (Optional[<a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/data/packed_sequence.py">PackedSequenceSpecs</a>] = None)</summary>
  <blockquote>

  ```sh
  packed_sequence_size: int = -1 # If a positive integer, this arg enables training with sequence packing and specifies the pack size
  tokenizer_model_name: str = None # Keep track of tokenizer model name, since each tokenizer produces a different packed sequence dataset file. This field is set by llm.finetune api
  packed_train_data_path: str = None # If specified, use this file for the packed training dataset instead of the default path
  packed_val_data_path: str = None # If specified, use this file for the packed validation dataset instead of the default path
  ```
  </blockquote>
  </details>
  </blockquote>
  </details>

  </blockquote>

</details>

<details>
  <summary>log (<a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/nemo_logger.py">NeMoLogger</a>)</summary>

  <blockquote>

  ```sh
  log_dir: Optional[str] = None # Directory to save logs
  log_local_rank_0_only: bool = False # Log only on local rank 0
  log_global_rank_0_only: bool = False # Log only on global rank 0
  ```
  <details><summary>ckpt (Optional[<a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/callbacks/model_checkpoint.py">ModelCheckpoint</a>] = None)</summary>
  <blockquote>      

  ```sh
  save_last: Optional[bool] = True # saves a `*-last` copy whenever a checkpoint file gets saved
  save_top_k: int = 3 # saves the top-k checkpoints according to 'monitor'
  every_n_epochs: int = None # Number of epochs between checkpoints
  every_n_train_steps: Optional[int] = None # Number of train steps between checkpoints
  save_on_train_epoch_end: Optional[bool] = False # Whether to run checkpointing at the end of the training epoch
  train_time_interval: Optional[timedelta] = None # After each interval, monitor checkpoints. Not to be used with 'every_n_epochs' or 'every_n_train_steps'
  ```
  </blockquote>
  </details>

  <details><summary>tensorboard (Optional[TensorBoardLogger] = None)</summary>
  <blockquote>

  ```sh
  save_dir: Union[str, Path] # Directory where tensorbroad log file will be saved
  name: Optional[str] = "lightning_logs" # Experiment name
  ```
  </blockquote>
  </details>

  <details><summary>wandb: Optional[WandbLogger] = None</summary>
  <blockquote>

  ```sh
  name: Optional[str] = None # Display name for the run
  project: Optional[str] = None # The name of the project to which this run will belong. If not set, the environment variable 'WANDB_PROJECT' will be used as a fallback. If both are not set, it defaults to 'lightning_logs'
  config: Dict # Add other config parameters, wandb_logger.experiment.config["key"] = value, wandb_logger.experiment.config.update({key1: val1, key2: val2}), wandb.config["key"] = value
  ```
  </blockquote>
  </details>

  </blockquote>

</details>

<details>
  <summary>optim (<a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/optim/base.py">OptimizerModule</a>) # Use either MegatronOptimizerModule or PytorchOptimizerModule</summary>

  <blockquote>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/optim/megatron.py">MegatronOptimizerModule</a></summary>
  <blockquote>

  <details><summary>config (<a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py">OptimizerConfig</a>)</summary>
  <blockquote>

  ```sh
  optimizer: str = 'adam' # Optimizer to use (one of Adam or SGD)
  lr: Optional[float] = None # Initial learning rate. Depending on decay style and initial warmup, the learning rate at each iteration would be different
  weight_decay: float = 0.01 # Weight decay coefficient for L2 regularization
  bf16: bool = False # If true, train with bf16 mixed precision training
  fp16: bool = False # If true, train with fp8 mixed precision training
  adam_beta1: float = 0.9 # First coefficient for computing running averages of gradient and its square in Adam optimizer
  adam_beta2: float = 0.999 # Second coefficient for computing running averages of gradient and its square in Adam optimizer
  adam_eps: float = 1e-08 # Term added to the denominator to improve numerical stability in Adam optimizer
  use_distributed_optimizer: bool = False # Distribute optimizer state over data-parallel replicas; this overrides same config in DistributedDataParallelConfig
  clip_grad: float 1.0 # Gradient clipping based on global L2 norm
  ```
  </blockquote>
  </details>
      
  <details><summary>lr_scheduler (Optional[CosineAnnealingScheduler[<a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/optim/base.py">LRSchedulerModule</a>]] = None)</summary>
  <blockquote>

  ```sh
  warmup_steps: int = 750 # Number of training steps in warmup stage
  constant_steps: int = 80000 # Number of steps to keep lr constant at
  min_lr: float = 6e-5 # Minimum lr to hold the learning rate after decay
  ```
  </blockquote>
  </details>

  </blockquote>
  </details>

  <details><summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/pytorch/optim/pytorch.py">PytorchOptimizerModule</a></summary>
  <blockquote>

  ```sh
  optimizer_fn: torch.optim.Optimizer # Eg. torch.optim.Adam
  lr_scheduler: Optional[LRSchedulerModule] = None # The learning rate scheduler module
  ```
  </blockquote>
  </details>

  </blockquote>

</details>

<details>
  <summary>resume (<a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/lightning/resume.py">AutoResume</a>)</summary>

  <blockquote>

  ```sh
  restore_config: Optional[RestoreConfig] = None # Optional config for selectively restoring specific parts like model weights, optimizer states, etc. If the config contains a path from HF or another non-NeMo checkpoint format, the checkpoint will be automatically converted to a NeMo compatible format
  resume_from_directory: Optional[str] = None # Path to the checkpointing directory to restore from; this takes precedence over 'restore_config'
  resume_from_path: Optional[str] = None # Path to a specific checkpoint to restore from
  adapter_path: Optional[str] = None # Path to any adapter checkpoints
  resume_if_exists: bool = False # Whether this experiment is resuming from a previous run. If True, it sets trainer._checkpoint_connector._ckpt_path so that the trainer should auto-resume
  ```
  </blockquote>

</details>

<details>
  <summary><a href="https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/peft/lora.py">LoRA</a> (PEFT)</summary>

  <blockquote>

  ```sh
  target_modules: List[str] = field(
      default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
  ) # A list of module names to apply LoRA to
  dim: int = 32 # Dimension of the low-rank projection space
  alpha: int = 32 # Weighting factor for the low-rank projection
  dropout: float = 0.0 # Dropout rate for the low-rank projection
  dropout_position: Literal['pre', 'post'] = 'post' # Position for applying dropout
  ```
  </blockquote>

</details>

</blockquote>
</details>