#!/usr/bin/env python3
"""
Modular and configurable training script for LoRA fine-tuning with NeMo.

This script provides a flexible framework for training language models using
LoRA (Low-Rank Adaptation) or full fine-tuning approaches. It supports both
local and SLURM execution environments.

Usage:
    python train.py --config config.yaml
    python train.py --model-name mistralai/Mistral-7B-Instruct-v0.3 --dataset-name rajpurkar/squad
    python train.py --use-slurm --account my_account --partition gpu --nodes 2
"""

import logging
import os
import sys
from typing import Optional

import nemo_run as run

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TrainingConfig, create_argument_parser, load_config
from executors import create_executor
from recipe_factory import create_recipe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_config(config: TrainingConfig) -> None:
    """
    Validate the training configuration for common issues.
    
    Args:
        config: Training configuration to validate
        
    Raises:
        ValueError: If configuration has validation errors
    """
    errors = []
    
    # Check required paths exist or can be created
    try:
        os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(config.paths.data_dir, exist_ok=True)
        if config.model.cache_dir:
            os.makedirs(config.model.cache_dir, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create required directories: {e}")
    
    # Validate SLURM configuration if needed
    if config.compute.use_slurm:
        required_slurm_fields = [
            'account', 'partition', 'remote_job_dir', 'nodes', 'gpus_per_node'
        ]
        for field in required_slurm_fields:
            if not getattr(config.compute, field):
                errors.append(f"SLURM configuration missing required field: {field}")
        
        # Additional validation for SSH tunnel
        if config.compute.tunnel_type.lower() == "ssh":
            ssh_required_fields = ['user', 'host']
            for field in ssh_required_fields:
                if not getattr(config.compute, field):
                    errors.append(f"SSH tunnel configuration missing required field: {field}")
        elif config.compute.tunnel_type.lower() not in ["ssh", "local"]:
            errors.append(f"Invalid tunnel type: {config.compute.tunnel_type}. Must be 'ssh' or 'local'")
    
    # Validate model and data configuration
    if not config.model.name:
        errors.append("Model name is required")
    
    if not config.data.dataset_name:
        errors.append("Dataset name is required")
    
    # Validate training parameters
    if config.trainer.max_steps <= 0:
        errors.append("max_steps must be positive")
    
    if config.optimizer.lr <= 0:
        errors.append("Learning rate must be positive")
    
    if config.lora.dim <= 0:
        errors.append("LoRA dimension must be positive")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


def setup_environment(config: TrainingConfig) -> None:
    """
    Setup environment variables and paths for training.
    
    Args:
        config: Training configuration
    """
    # Set environment variables
    env_vars = config.environment.to_dict()
    env_vars.update({
        "HF_HOME": config.model.cache_dir,
        "HF_TOKEN": config.model.token,
        "TRITON_CACHE_DIR": config.model.cache_dir
    })
    
    for key, value in env_vars.items():
        if value is not None:
            os.environ[key] = str(value)
    
    logger.info(f"Environment configured with {len(env_vars)} variables")


def run_training(config: TrainingConfig, recipe_type: str = "lora", dry_run: bool = False) -> None:
    """
    Execute the training run with the given configuration.
    
    Args:
        config: Training configuration
        recipe_type: Type of training recipe ("lora" or "full")
        dry_run: If True, validate configuration but don't run training
    """
    logger.info(f"Starting {recipe_type} training with experiment: {config.experiment_name}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Compute: {config.compute.nodes} nodes, {config.compute.gpus_per_node} GPUs/node")
    
    # Validate configuration
    validate_config(config)
    logger.info("Configuration validation passed")
    
    # Setup environment
    setup_environment(config)
    
    if dry_run:
        logger.info("Dry run completed successfully")
        return
    
    # Create training recipe
    logger.info(f"Creating {recipe_type} training recipe")
    recipe = create_recipe(config, recipe_type)

    # Create executor
    executor_type = "SLURM" if config.compute.use_slurm else "Local"
    logger.info(f"Creating {executor_type} executor")
    executor = create_executor(config)
    
    # Run training experiment
    logger.info("Starting training experiment")
    with run.Experiment(config.experiment_name) as exp:
        exp.add(recipe, executor=executor, name=f"{recipe_type}_training")
        exp.run(sequential=True, tail_logs=True)
    
    logger.info("Training completed successfully")


def save_config_template(output_path: str = "config_template.yaml") -> None:
    """
    Save a template configuration file for reference.
    
    Args:
        output_path: Path where to save the template
    """
    config = TrainingConfig()
    config.to_yaml(output_path)
    logger.info(f"Configuration template saved to: {output_path}")


def main():
    """Main entry point for the training script."""
    parser = create_argument_parser()
    
    # Add additional arguments specific to training script
    parser.add_argument('--recipe-type', choices=['lora', 'full'], default='lora',
                       help='Type of training recipe to use')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running training')
    parser.add_argument('--save-config-template', type=str, metavar='PATH',
                       help='Save a configuration template to the specified path and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle template generation
    if args.save_config_template:
        save_config_template(args.save_config_template)
        return
    
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config(args)
        
        # Always show key configuration values (not just in verbose mode)
        logger.info("Configuration summary:")
        logger.info(f"  Experiment: {config.experiment_name}")
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Dataset: {config.data.dataset_name}")
        logger.info(f"  Max steps: {config.trainer.max_steps}")
        logger.info(f"  Learning rate: {config.optimizer.lr}")
        logger.info(f"  LoRA dim: {config.lora.dim}")
        logger.info(f"  Batch size: {config.data.micro_batch_size}")
        logger.info(f"  Sequence length: {config.data.seq_length}")
        
        if args.verbose:
            logger.debug("Detailed configuration:")
            logger.debug(f"  LoRA target modules: {config.lora.target_modules}")
            logger.debug(f"  LoRA dropout: {config.lora.dropout}")
            logger.debug(f"  Warmup steps: {config.optimizer.warmup_steps}")
            logger.debug(f"  Weight decay: {config.optimizer.weight_decay}")
            logger.debug(f"  Data split: {config.data.split}")
            logger.debug(f"  Compute: {config.compute.nodes} nodes, {config.compute.gpus_per_node} GPUs")
            logger.debug(f"  Use SLURM: {config.compute.use_slurm}")
            logger.debug(f"  Checkpoint dir: {config.paths.checkpoint_dir}")
        
        # Run training
        run_training(config, args.recipe_type, args.dry_run)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 