import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    
    def __post_init__(self):
        if self.token is None:
            self.token = os.getenv("HF_TOKEN")
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.getcwd(), "models/hf_cache")


@dataclass
class DataConfig:
    """Configuration for dataset parameters."""
    dataset_name: str = "rajpurkar/squad"
    seq_length: int = 4096
    micro_batch_size: int = 32
    split: str = "train[:100]"
    tokenizer_name: Optional[str] = None
    
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.dataset_name


@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters."""
    target_modules: List[str] = field(default_factory=lambda: ['o_proj'])
    dim: int = 32
    dropout: float = 0.1
    lora_A_init_method: str = 'xavier'
    lora_B_init_method: str = 'zero'


@dataclass
class OptimizerConfig:
    """Configuration for optimizer parameters."""
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple = field(default_factory=lambda: (0.9, 0.95))
    warmup_steps: int = 100
    constant_steps: int = 0
    min_lr: float = 2e-5


@dataclass
class TrainerConfig:
    """Configuration for trainer parameters."""
    max_steps: int = 10
    num_sanity_val_steps: int = 0
    val_check_interval: int = 1
    log_every_n_steps: int = 1
    checkpoint_filename: str = "LoRA_Finetune"
    version: int = 1

@dataclass
class ComputeConfig:
    """Configuration for compute resources."""
    nodes: int = 1
    gpus_per_node: int = 8
    time: str = "04:00:00"
    use_slurm: bool = False
    
    # SLURM-specific configurations
    user: Optional[str] = None
    host: Optional[str] = None
    remote_job_dir: Optional[str] = None
    account: Optional[str] = None
    partition: Optional[str] = None
    container_image: str = "nvcr.io/nvidia/nemo:25.07"
    custom_mounts: Optional[List[str]] = None
    retries: int = 0
    tunnel_type: str = "ssh"  # Options: "ssh" or "local"


@dataclass
class PathConfig:
    """Configuration for file paths."""
    project_root: str = field(default_factory=lambda: os.path.join(
        "/workspace", "LoRA_Train"
    ))
    checkpoint_dir: Optional[str] = None
    data_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.project_root, "models/checkpoints")
        if self.data_dir is None:
            self.data_dir = os.path.join(self.project_root, "data")


@dataclass
class EnvironmentConfig:
    """Configuration for environment variables."""
    transformers_offline: str = "0"
    torch_nccl_avoid_record_streams: str = "1"
    nccl_nvls_enable: str = "0"
    nvte_dp_amax_reduce_interval: str = "0"
    nvte_async_amax_reduction: str = "1"
    nvte_fused_attn: str = "0"
    custom_env_vars: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for use with executors."""
        env_vars = {
            "TRANSFORMERS_OFFLINE": self.transformers_offline,
            "TORCH_NCCL_AVOID_RECORD_STREAMS": self.torch_nccl_avoid_record_streams,
            "NCCL_NVLS_ENABLE": self.nccl_nvls_enable,
            "NVTE_DP_AMAX_REDUCE_INTERVAL": self.nvte_dp_amax_reduce_interval,
            "NVTE_ASYNC_AMAX_REDUCTION": self.nvte_async_amax_reduction,
            "NVTE_FUSED_ATTN": self.nvte_fused_attn,
        }
        
        if self.custom_env_vars:
            env_vars.update(self.custom_env_vars)
            
        return env_vars


@dataclass
class TrainingConfig:
    """Main configuration class that combines all configuration components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    experiment_name: str = "LoRA_Finetune_Experiment"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace, base_config: Optional['TrainingConfig'] = None) -> 'TrainingConfig':
        """Create configuration from command line arguments, optionally updating a base config."""
        # Start with base config if provided, otherwise create new config
        config = base_config if base_config is not None else cls()
        
        # Update model config (only if explicitly provided)
        if hasattr(args, 'model_name') and args.model_name:
            config.model.name = args.model_name
        if hasattr(args, 'hf_token') and args.hf_token:
            config.model.token = args.hf_token
        if hasattr(args, 'model_cache_dir') and args.model_cache_dir:
            config.model.cache_dir = args.model_cache_dir
            
        # Update data config (only if explicitly provided)
        if hasattr(args, 'dataset_name') and args.dataset_name:
            config.data.dataset_name = args.dataset_name
        if hasattr(args, 'seq_length') and args.seq_length:
            config.data.seq_length = args.seq_length
        if hasattr(args, 'micro_batch_size') and args.micro_batch_size:
            config.data.micro_batch_size = args.micro_batch_size
        if hasattr(args, 'split') and args.split:
            config.data.split = args.split
            
        # Update LoRA config (only if explicitly provided)
        if hasattr(args, 'lora_dim') and args.lora_dim:
            config.lora.dim = args.lora_dim
        if hasattr(args, 'lora_dropout') and args.lora_dropout is not None:
            config.lora.dropout = args.lora_dropout
        if hasattr(args, 'target_modules') and args.target_modules:
            config.lora.target_modules = args.target_modules
            
        # Update optimizer config (only if explicitly provided)
        if hasattr(args, 'learning_rate') and args.learning_rate:
            config.optimizer.lr = args.learning_rate
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            config.optimizer.weight_decay = args.weight_decay
        if hasattr(args, 'warmup_steps') and args.warmup_steps is not None:
            config.optimizer.warmup_steps = args.warmup_steps
            
        # Update trainer config (only if explicitly provided)
        if hasattr(args, 'max_steps') and args.max_steps:
            config.trainer.max_steps = args.max_steps
        if hasattr(args, 'checkpoint_filename') and args.checkpoint_filename:
            config.trainer.checkpoint_filename = args.checkpoint_filename
        if hasattr(args, 'version') and args.version:
            config.trainer.version = args.version
        # Update compute config (only if explicitly provided)
        if hasattr(args, 'nodes') and args.nodes:
            config.compute.nodes = args.nodes
        if hasattr(args, 'gpus_per_node') and args.gpus_per_node:
            config.compute.gpus_per_node = args.gpus_per_node
        if hasattr(args, 'time') and args.time:
            config.compute.time = args.time
        if hasattr(args, 'use_slurm') and args.use_slurm is not None:
            config.compute.use_slurm = args.use_slurm
        
        # SLURM-specific args (only if explicitly provided)
        for attr in ['user', 'host', 'remote_job_dir', 'account', 'partition', 'container_image']:
            if hasattr(args, attr) and getattr(args, attr):
                setattr(config.compute, attr, getattr(args, attr))
        
        # Handle tunnel_type specifically
        if hasattr(args, 'tunnel_type') and args.tunnel_type:
            config.compute.tunnel_type = args.tunnel_type
                
        # Update paths config (only if explicitly provided)
        if hasattr(args, 'project_root') and args.project_root:
            config.paths.project_root = args.project_root
        if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
            config.paths.checkpoint_dir = args.checkpoint_dir
        if hasattr(args, 'data_dir') and args.data_dir:
            config.paths.data_dir = args.data_dir
            
        # Update experiment name (only if explicitly provided)
        if hasattr(args, 'experiment_name') and args.experiment_name:
            config.experiment_name = args.experiment_name
            
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'lora' in data:
            config.lora = LoRAConfig(**data['lora'])
        if 'optimizer' in data:
            config.optimizer = OptimizerConfig(**data['optimizer'])
        if 'trainer' in data:
            config.trainer = TrainerConfig(**data['trainer'])
        if 'compute' in data:
            config.compute = ComputeConfig(**data['compute'])
        if 'paths' in data:
            config.paths = PathConfig(**data['paths'])
        if 'environment' in data:
            config.environment = EnvironmentConfig(**data['environment'])
        if 'experiment_name' in data:
            config.experiment_name = data['experiment_name']
            
        return config
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': {
                'name': self.model.name,
                'cache_dir': self.model.cache_dir,
                'token': self.model.token,
            },
            'data': {
                'dataset_name': self.data.dataset_name,
                'seq_length': self.data.seq_length,
                'micro_batch_size': self.data.micro_batch_size,
                'split': self.data.split,
                'tokenizer_name': self.data.tokenizer_name,
            },
            'lora': {
                'target_modules': self.lora.target_modules,
                'dim': self.lora.dim,
                'dropout': self.lora.dropout,
                'lora_A_init_method': self.lora.lora_A_init_method,
                'lora_B_init_method': self.lora.lora_B_init_method,
            },
            'optimizer': {
                'lr': self.optimizer.lr,
                'weight_decay': self.optimizer.weight_decay,
                'betas': self.optimizer.betas,
                'warmup_steps': self.optimizer.warmup_steps,
                'constant_steps': self.optimizer.constant_steps,
                'min_lr': self.optimizer.min_lr,
            },
            'trainer': {
                'max_steps': self.trainer.max_steps,
                'num_sanity_val_steps': self.trainer.num_sanity_val_steps,
                'val_check_interval': self.trainer.val_check_interval,
                'log_every_n_steps': self.trainer.log_every_n_steps,
                'checkpoint_filename': self.trainer.checkpoint_filename,
                'version': self.trainer.version,
            },
            'compute': {
                'nodes': self.compute.nodes,
                'gpus_per_node': self.compute.gpus_per_node,
                'time': self.compute.time,
                'use_slurm': self.compute.use_slurm,
                'user': self.compute.user,
                'host': self.compute.host,
                'remote_job_dir': self.compute.remote_job_dir,
                'account': self.compute.account,
                'partition': self.compute.partition,
                'container_image': self.compute.container_image,
                'custom_mounts': self.compute.custom_mounts,
                'retries': self.compute.retries,
                'tunnel_type': self.compute.tunnel_type,
            },
            'paths': {
                'project_root': self.paths.project_root,
                'checkpoint_dir': self.paths.checkpoint_dir,
                'data_dir': self.paths.data_dir,
            },
            'environment': {
                'transformers_offline': self.environment.transformers_offline,
                'torch_nccl_avoid_record_streams': self.environment.torch_nccl_avoid_record_streams,
                'nccl_nvls_enable': self.environment.nccl_nvls_enable,
                'nvte_dp_amax_reduce_interval': self.environment.nvte_dp_amax_reduce_interval,
                'nvte_async_amax_reduction': self.environment.nvte_async_amax_reduction,
                'nvte_fused_attn': self.environment.nvte_fused_attn,
                'custom_env_vars': self.environment.custom_env_vars,
            },
            'experiment_name': self.experiment_name,
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser for all configuration options."""
    parser = argparse.ArgumentParser(
        description='Configurable LoRA Fine-tuning with NeMo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model-name mistralai/Mistral-7B-Instruct-v0.3 --dataset-name rajpurkar/squad
  python train.py --config config.yaml --max-steps 100
  python train.py --use-slurm --nodes 2 --account my_account --partition gpu
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model-name', type=str, help='Hugging Face model name')
    model_group.add_argument('--hf-token', type=str, help='Hugging Face token')
    model_group.add_argument('--model-cache-dir', type=str, help='Model cache directory')
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--dataset-name', type=str, help='Dataset name')
    data_group.add_argument('--seq-length', type=int, help='Sequence length')
    data_group.add_argument('--micro-batch-size', type=int, help='Micro batch size')
    data_group.add_argument('--split', type=str, help='Dataset split (e.g., "train[:100]")')
    
    # LoRA configuration
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument('--lora-dim', type=int, help='LoRA dimension/rank')
    lora_group.add_argument('--lora-dropout', type=float, help='LoRA dropout rate')
    lora_group.add_argument('--target-modules', nargs='+', help='Target modules for LoRA')
    
    # Optimizer configuration
    optim_group = parser.add_argument_group('Optimizer Configuration')
    optim_group.add_argument('--learning-rate', type=float, help='Learning rate')
    optim_group.add_argument('--weight-decay', type=float, help='Weight decay')
    optim_group.add_argument('--warmup-steps', type=int, help='Warmup steps')
    
    # Trainer configuration
    trainer_group = parser.add_argument_group('Trainer Configuration')
    trainer_group.add_argument('--max-steps', type=int, help='Maximum training steps')
    trainer_group.add_argument('--checkpoint-filename', type=str, help='Checkpoint filename')
    trainer_group.add_argument('--version', type=int, help='Version')
    # Compute configuration
    compute_group = parser.add_argument_group('Compute Configuration')
    compute_group.add_argument('--nodes', type=int, help='Number of nodes')
    compute_group.add_argument('--gpus-per-node', type=int, help='GPUs per node')
    compute_group.add_argument('--time', type=str, help='Job time limit (HH:MM:SS)')
    compute_group.add_argument('--use-slurm', action='store_true', help='Use SLURM executor')
    
    # SLURM configuration
    slurm_group = parser.add_argument_group('SLURM Configuration')
    slurm_group.add_argument('--user', type=str, help='SLURM user')
    slurm_group.add_argument('--host', type=str, help='SLURM host')
    slurm_group.add_argument('--remote-job-dir', type=str, help='Remote job directory')
    slurm_group.add_argument('--account', type=str, help='SLURM account')
    slurm_group.add_argument('--partition', type=str, help='SLURM partition')
    slurm_group.add_argument('--container-image', type=str, help='Container image')
    slurm_group.add_argument('--tunnel-type', choices=['ssh', 'local'], default='ssh',
                           help='Type of tunnel to use for SLURM jobs (ssh or local)')
    
    # Path configuration
    path_group = parser.add_argument_group('Path Configuration')
    path_group.add_argument('--project-root', type=str, help='Project root directory')
    path_group.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    path_group.add_argument('--data-dir', type=str, help='Data directory')
    
    # Experiment configuration
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    
    return parser


def load_config(args: argparse.Namespace) -> TrainingConfig:
    """Load configuration from various sources with proper precedence."""
    # Start with base configuration from file if provided
    if hasattr(args, 'config') and args.config:
        # Load from file if specified
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            base_config = TrainingConfig.from_yaml(args.config)
        elif args.config.endswith('.json'):
            base_config = TrainingConfig.from_json(args.config)
        else:
            raise ValueError(f"Unsupported config file format: {args.config}")
    else:
        # Use default configuration if no file specified
        base_config = TrainingConfig()
    
    # Override with command line arguments (only those explicitly provided)
    final_config = TrainingConfig.from_args(args, base_config)
    
    return final_config 