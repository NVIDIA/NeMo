"""
Executor utilities for running training jobs locally or on SLURM.
"""

import os
from typing import Dict, List, Optional

import nemo_run as run
from config import TrainingConfig


def create_local_executor(config: TrainingConfig) -> run.LocalExecutor:
    """
    Create a local executor for running training jobs on local machine.
    
    Args:
        config: Training configuration object
        
    Returns:
        LocalExecutor configured for the training job
    """
    env_vars = config.environment.to_dict()
    env_vars.update({
        "HF_HOME": config.model.cache_dir,
        "HF_TOKEN": config.model.token,
        "TRITON_CACHE_DIR": config.model.cache_dir
    })
    
    executor = run.LocalExecutor(
        ntasks_per_node=config.compute.gpus_per_node, 
        launcher="torchrun", 
        env_vars=env_vars
    )
    
    return executor


def create_slurm_executor(config: TrainingConfig) -> run.SlurmExecutor:
    """
    Create a SLURM executor for running training jobs on a SLURM cluster.
    
    Args:
        config: Training configuration object
        
    Returns:
        SlurmExecutor configured for the training job
        
    Raises:
        RuntimeError: If required SLURM configuration is missing
    """
    if not all([
        config.compute.remote_job_dir, 
        config.compute.account, 
        config.compute.partition, 
        config.compute.nodes, 
        config.compute.gpus_per_node
    ]):
        raise RuntimeError(
            "Please set remote_job_dir, account, partition, nodes, and gpus_per_node "
            "in compute configuration for using SLURM executor."
        )
    
    # Prepare mounts
    mounts = []
    if config.compute.custom_mounts:
        mounts.extend(config.compute.custom_mounts)
    
    # Prepare environment variables
    env_vars = config.environment.to_dict()
    env_vars.update({
        "HF_HOME": config.model.cache_dir,
        "HF_TOKEN": config.model.token,
        "TRITON_CACHE_DIR": config.model.cache_dir
    })
    
    # Create tunnel based on configuration
    if config.compute.tunnel_type.lower() == "ssh":
        tunnel = run.SSHTunnel(
            user=config.compute.user, 
            host=config.compute.host, 
            job_dir=config.compute.remote_job_dir
        )
    elif config.compute.tunnel_type.lower() == "local":
        tunnel = run.LocalTunnel(
            job_dir=config.compute.remote_job_dir
        )
    else:
        raise ValueError(f"Unsupported tunnel type: {config.compute.tunnel_type}. Choose 'ssh' or 'local'.")
    
    # Create packager to include necessary files
    current_dir = os.getcwd()
    packager = run.PatternPackager(
        include_pattern=[
            os.path.join(current_dir, "train.py"),
            os.path.join(current_dir, "config.py"),
            os.path.join(current_dir, "executors.py"),
            os.path.join(current_dir, "recipe_factory.py"),
            os.path.join(current_dir, "data_modules.py"),
        ],
        relative_path=[current_dir] * 5
    )
    
    # Create executor
    executor = run.SlurmExecutor(
        account=config.compute.account,
        partition=config.compute.partition,
        tunnel=tunnel,
        nodes=config.compute.nodes,
        ntasks_per_node=config.compute.gpus_per_node,
        gpus_per_node=config.compute.gpus_per_node,
        mem="0",
        exclusive=True,
        gres=f"gpu:{config.compute.gpus_per_node}",
        packager=packager,
    )
    
    # Configure additional properties
    executor.container_image = config.compute.container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = config.compute.retries
    executor.time = config.compute.time
    
    return executor


def create_executor(config: TrainingConfig) -> run.Executor:
    """
    Create appropriate executor based on configuration.
    
    Args:
        config: Training configuration object
        
    Returns:
        Configured executor (Local or SLURM)
    """
    if config.compute.use_slurm:
        return create_slurm_executor(config)
    else:
        return create_local_executor(config) 