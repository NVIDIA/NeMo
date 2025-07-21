import os
import json
import logging
from typing import Dict, Any, List, Optional
from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from nemo.collections.llm.tools.autotuner.core.utils import validate_all_configs, _load_args_from_config_dir

import nemo_run as run

logger = logging.getLogger(__name__)


def lepton_executor(
    nodes: int, 
    devices: int,
    resource_shape: str = "gpu.8xh200",
    container_image: str = "nvcr.io/nvidia/nemo:25.02",
    nemo_run_dir: str = "/nemo-workspace/nemo-run",
    mount_path: str = "/nemo-workspace",
    mount_from: str = "node-nfs:shared",
    node_group: str = "nebius-h200-01",
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    torch_home: str = "/nemo-workspace/.cache",
    pythonpath: str = "/nemo-workspace/nemo-run:$PYTHONPATH"
) -> run.LeptonExecutor:
    """Create a Lepton executor for training with dynamic configuration."""
    mounts = [{
        "path": "/",
        "mount_path": mount_path,
        "from": mount_from
    }]
    env_vars = {
        "PYTHONPATH": pythonpath,
        "TORCH_HOME": torch_home,
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key

    return run.LeptonExecutor(
        resource_shape=resource_shape,
        container_image=container_image,
        nemo_run_dir=nemo_run_dir,
        mounts=mounts,
        node_group=node_group,
        nodes=nodes,
        nprocs_per_node=devices,
        env_vars=env_vars,
        launcher="torchrun",
    )

def run_pretraining(
    base_config, 
    configs: Dict, 
    base_config_matches: List[str] = None, 
    sequential: bool = False,
    executor_config: Dict[str, Any] = None,
    memory_analysis: Dict[str, Dict[str, Any]] = None,
    run_all: bool = False
):
    """Run pretraining only without results collection."""
    logger.info("Starting AutoTune pretraining...")
    
    if base_config_matches is None:
        base_config_matches = []
    if executor_config is None:
        executor_config = {}
    if memory_analysis is None:
        memory_analysis = {}

    configs_to_run = {}
    skipped_configs = {}
    base_config_will_run = True

    base_analysis = memory_analysis.get("base_config", {})
    base_will_oom = base_analysis.get("will_oom", False)
    if base_will_oom and not run_all:
        base_config_will_run = False
        skipped_configs["base_config"] = "Potential CUDA OOM"
        logger.warning("Skipping base_config due to potential CUDA OOM (use --run-all to force)")

    for config_name, config_obj in configs.items():
        analysis = memory_analysis.get(config_name, {})
        will_oom = analysis.get("will_oom", False)
        if will_oom and not run_all:
            skipped_configs[config_name] = "Potential CUDA OOM"
            logger.warning(f"Skipping {config_name} due to potential CUDA OOM (use --run-all to force)")
        else:
            configs_to_run[config_name] = config_obj

    total_configs = len(configs) + (1 if not base_config_matches else 0)
    configs_to_run_count = len(configs_to_run) + (1 if base_config_will_run and not base_config_matches else 0)
    skipped_count = len(skipped_configs)

    logger.info(f"Configuration filtering summary:")
    logger.info(f"  Total configurations: {total_configs}")
    logger.info(f"  Configurations to run: {configs_to_run_count}")
    logger.info(f"  Skipped configurations: {skipped_count}")

    if configs_to_run_count == 0:
        logger.error("No configurations to run! All were filtered out due to potential OOM.")
        logger.error("Use --run-all flag to run anyway, or adjust your configuration parameters.")
        return {
            'total_configs': total_configs,
            'configs_run': 0,
            'configs_skipped': skipped_count,
            'skipped_configs': skipped_configs,
            'status': 'no_configs_to_run'
        }

    logger.info("Executor Settings...")
    logger.info(executor_config)

    executor = lepton_executor(
        nodes=base_config.trainer.num_nodes,
        devices=base_config.trainer.devices,
        **executor_config
    )

    logger.info("Running filtered configurations...")

    with run.Experiment("pretrain-magic") as exp:
        if not base_config_matches and base_config_will_run:
            exp.add(base_config, executor=executor, name="base_config")
            logger.info("Added base_config to experiment")
        elif not base_config_matches and not base_config_will_run:
            logger.info("Skipped base_config due to potential CUDA OOM")
        else:
            logger.info(f"Skipping base_config as it matches: {', '.join(base_config_matches)}")
        
        idx = 1
        for config_name, recipe in configs_to_run.items():
            if config_name in base_config_matches:
                exp.add(recipe, executor=executor, name=f'base-config')
                logger.info(f"Added {config_name} as base_config_equivalent (matches base config)")
            else:
                exp.add(recipe, executor=executor, name=f'config-{idx}')
                logger.info(f"Added {config_name} as config-{idx}")
                idx = idx + 1

        exp.run(sequential=sequential)

    logger.info("AutoTune pretraining completed successfully!")
    if base_config_matches:
        logger.info(f"Note: Base config was not run separately as it matches {len(base_config_matches)} generated config(s)")
    if skipped_count > 0:
        logger.info(f"Note: {skipped_count} configuration(s) were skipped due to potential CUDA OOM")

    return {
        'total_configs': total_configs,
        'configs_run': configs_to_run_count,
        'configs_skipped': skipped_count,
        'skipped_configs': skipped_configs,
        'status': 'completed'
    }
