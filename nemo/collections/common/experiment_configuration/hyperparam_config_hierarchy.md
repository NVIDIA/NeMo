<details open>
<summary>recipe</summary>

  - <details open>
    <summary>trainer (pytorch_lightning.Trainer)</summary>
    
      - <details open>
        <summary>strategy (MegatronStrategy)</summary>

        -  tensor_model_parallel_size: int = 1,
        -  pipeline_model_parallel_size: int = 1
        -  virtual_pipeline_model_parallel_size: Optional[int] = None
        -  context_parallel_size: int = 1
        -  sequence_parallel: bool = False
        -  expert_model_parallel_size: int = 1
        -  pipeline_dtype: Optional[torch.dtype] = None,
        - <details>    
          <summary>ddp: Union[DDPLiteral, DistributedDataParallelConfig] = "megatron"</summary>

            -  grad_reduce_in_fp32: bool = False
            -  overlap_grad_reduce: bool = False
            -  overlap_param_gather: bool = False
            -  align_param_gather: bool = False
            -  use_distributed_optimizer: bool = False
            -  check_for_nan_in_grad: bool = False
            -  bucket_size: Optional[int] = None
            -  average_in_collective: bool = False
            -  fp8_param_gather: bool = False    
          </details>
        </details>

      - <details open>
        <summary>callbacks (Optional[List[Callback]]=None)</summary>

          - <details>    
            <summary>TimingCallback (Callback)</summary>

            - reduction: str = "mean"
            - sync_cuda: bool = False
            - buffer_size: int = 1
            </details>

          - <details>
            <summary>MegatronCommOverlapCallback (Callback)</summary>

            - tp_comm_overlap: bool = None
            - tp_comm_overlap_cfg: TransformerLayerTPOverlapCfg = None
            - overlap_p2p_comm: bool = None
            - batch_p2p_comm: bool = None
            - overlap_grad_reduce: bool = None
            - overlap_param_gather: bool = None
            - overlap_param_gather_with_optimizer_step: bool = None
            - align_param_gather: bool = None
            - bucket_size: int = None
            - defer_embedding_wgrad_compute: bool = None
            - wgrad_deferral_limit: int = None
            </details>

          - <details>
            <summary>NsysCallback (Callback)</summary>

            - start_step: int
            - end_step: int
            - ranks: List[int] = [0]
            - gen_shape: bool = False
            </details>

          - <details>
            <summary>MemoryProfileCallback (Callback)</summary>

            - dir: str = "/mem_profile"
            - warn_cycles: bool = True
            - ranks: List = []
            </details>

          - <details>
            <summary>GarbageCollectionCallback (Callback)</summary>

            - gc_interval_train: int
            - gc_interval_val: int
            </details>
        </details>

      - <details>
        <summary>plugins (nemo_run.Plugin)</summary>

          - <details>
            <summary>MegatronMixedPrecision</summary>

            - precision: Literal["16-mixed", "bf16-mixed", "32"]
            - params_dtype: torch.dtype = None
            - pipeline_dtype: torch.dtype = None
            - autocast_enabled: bool = False
            - grad_reduce_in_fp32: bool = True # this overrides same config in DistributedDataParallelConfig
            - fp8: str = None
            - fp8_margin: int = 0
            - fp8_amax_history_len: int = 1
            - fp8_amax_compute_algo: str = "most_recent"
            - fp8_params: bool = False
            </details>

          - <details>
            <summary>PerfEnvPlugin</summary>

            - enable_layernorm_sm_margin: bool = True
            - layernorm_sm_margin: int = 16
            - enable_vboost: bool = False
            </details>
        </details>
    </details>

  - <details>
    <summary>model (pytorch_lightning.LightningModule)</summary>

    - <details>
      <summary>config (GPTConfig)</summary>

      - seq_length: int = 1024
      - attention_softmax_in_fp32: bool = False
      - num_layers: int = 0
      - hidden_size: int = 0
      - num_attention_heads: int = 0
      - num_query_groups: Optional[int] = None
      - ffn_hidden_size: Optional[int] = None
      - hidden_dropout: float = 0.1
      - attention_dropout: float = 0.1
      - add_bias_linear: bool = True
      - gated_linear_unit: bool = False
      - activation_func: Callable = F.gelu
      - normalization: bool = "LayerNorm"
      - layernorm_epsilon: float = 1e-5
      - layernorm_zero_centered_gamma: bool = False
  
      - masked_softmax_fusion: bool = True
      - cross_entropy_loss_fusion: bool = True
      - gradient_accumulation_fusion: bool = _grad_accum_fusion_available # Requires the custom CUDA extension fused_weight_gradient_mlp_cuda module
      - bias_activation_fusion: bool = False
      - bias_dropout_fusion: bool = False 
      - apply_rope_fusion: bool = False
  
      - recompute_granularity: Optional[str] = None # Determines which type of activation recompute to use. If set, must be 'selective' or 'full'.
      - recompute_method: Optional[str] = None # Determines which transformer layers will be recomputed. If set, must be 'uniform' or 'block'.
      - recompute_num_layers: Optional[int] = None 
      distribute_saved_activations: Optional[bool] = None # If True, distribute recomputed activations across the - model parallel group.

      - enable_cuda_graph: bool = False # When set to true, TransformerLayer layers are swapped with a CUDA graphed version.
      - external_cuda_graph: bool = False # When set to true, TransformerLayer layers are swapped with user provided CUDA graphs.
      </details>
    </details>
  - <details>
    <summary>data (pytorch_lightning.LightningDataModule)</summary>

      - <details>
        <summary>MockDataModule</summary>
      
        - seq_length: int = 2048
        - tokenizer: Optional["TokenizerSpec"] = None
        - micro_batch_size: int = 4
        - global_batch_size: int = 8
        - rampup_batch_size: Optional[List[int]] = None
        - num_train_samples: int = 10_000
        - num_val_samples: int = 10_000
        - num_test_samples: int = 10_000
        - num_workers: int = 8
        - pin_memory: bool = True
        - persistent_workers: bool = False
        - create_attention_mask: bool = False
        </details>

      - <details>
        <summary>SquadDataModule (For fine-tuning jobs)</summary>

        - seq_length: int = 2048
        - tokenizer: Optional["TokenizerSpec"] = None
        - micro_batch_size: int = 4
        - global_batch_size: int = 8
        - <details>
          <summary>packed_sequence_specs (Optional["PackedSequenceSpecs"] = None)</summary>

          - packed_sequence_size: int = -1
          - tokenizer_model_name: str = None
          - packed_data_path: str = None
          </details>
        </details>
    </details>

  - <details>
    <summary>log (nemo.lightning.NeMoLogger)</summary>

      - log_dir: Optional[str] = None
      - log_local_rank_0_only: bool = False
      - log_global_rank_0_only: bool = False
      - <details>
        <summary>ckpt (Optional[ModelCheckpoint] = None)</summary>
        
        - save_last: Optional[bool] = True
        - save_top_k: int = 3
        - every_n_epochs: int = None
        - every_n_train_steps: Optional[int] = None
        - save_on_train_epoch_end: Optional[bool] = False
        - train_time_interval: Optional[timedelta] = None
        </details>
      - <details>
        <summary>tensorboard (Optional[TensorBoardLogger] = None)</summary>

        - save_dir: Union[str, Path]
        - name: Optional[str] = "lightning_logs"
        </details>
      - <details>
        <summary>wandb: Optional[WandbLogger] = None</summary>

        - name: Optional[str] = None
        - project: Optional[str] = None
        - config: Dict
        </details>
    </details>

  - <details open>
    <summary>optim (OptimizerModule) # Use either MegatronOptimizerModule or PytorchOptimizerModule</summary>

    - <details open>
      <summary>MegatronOptimizerModule</summary>

      - <details open>
        <summary>config (OptimizerConfig)</summary>

        - optimizer: str = 'adam'
        - lr: Optional[float] = None
        - weight_decay: float = 0.01
        - bf16: bool = False
        - fp16: bool = False
        - adam_beta1: float = 0.9
        - adam_beta2: float = 0.999
        - adam_eps: float = 1e-08
        - use_distributed_optimizer: bool = False # this overrides same config in DistributedDataParallelConfig
        - clip_grad: float 1.0
        </details>
      
      - <details>
        <summary>lr_scheduler (Optional[LRSchedulerModule] = None)</summary>

        - warmup_steps: int = 750
        - constant_steps: int = 80000
        - min_lr: float = 6e-5
        </details>
      </details>

    - <details>
      <summary>PytorchOptimizerModule</summary>

      - optim_cls # Eg. torch.optim.Adam
      - config: dict = {'lr': 3e-4}
      - lr_scheduler: Optional[LRSchedulerModule] = None
      </details>

    </details>

  - <details>
    <summary>resume (nemo.lightning.AutoResume)</summary>

    - restore_config: Optional[RestoreConfig] = None
    - resume_from_directory: Optional[str] = None
    - resume_from_path: Optional[str] = None
    - adapter_path: Optional[str] = None
    - resume_if_exists: bool = False
    - resume_past_end: bool = False
    - resume_ignore_no_checkpoint: bool = False
    </details>

  - <details>
    <summary>LoRA (PEFT)</summary>

    - target_modules: List[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    - dim: int = 32
    - alpha: int = 32
    - dropout: float = 0.0
    - dropout_position: Literal['pre', 'post'] = 'post'
    - lora_A_init_method: str = "xavier"
    - lora_B_init_method: str = "zero"
    </details>
</details>
