#!/usr/bin/env python3
"""
AutoTuner Launcher - Using NeMo Run Experiment approach
Uses run.Experiment with run.LeptonExecutor and run.Partial/run.Script for clean execution.
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import nemo_run as run
from nemo_run.config import Partial, Script



logger = logging.getLogger(__name__)

def create_lepton_executor(
    resource_shape: str = "cpu.small",
    container_image: str = "python:3.11",
    nemo_run_dir: str = "/nemo-workspace/nemo-run",
    mounts: Optional[list] = None,
    node_group: Optional[str] = None,
    nodes: int = 1,
    env_vars: Optional[Dict[str, str]] = None
) -> run.LeptonExecutor:
    """Create a LeptonExecutor with the specified configuration."""
    
    if mounts is None:
        mounts = []
    
    if env_vars is None:
        env_vars = {}
    
    # Add default environment variables
    default_env_vars = {
        "PYTHONPATH": "/tmp/NeMo:/tmp/Run:$PYTHONPATH",
        "TORCH_HOME": "/workspace/.cache",
    }
    env_vars.update(default_env_vars)
    
    return run.LeptonExecutor(
        resource_shape=resource_shape,
        container_image=container_image,
        nemo_run_dir=nemo_run_dir,
        mounts=mounts,
        node_group=node_group,
        nodes=nodes,
        nprocs_per_node=nprocs_per_node,
        env_vars=env_vars,
        shared_memory_size=1024,
    )

def setup_nemo_environment():
    """Setup NeMo environment in the remote container."""
    import subprocess
    import sys
    import os
    
    # Clone and install NeMo
    print("Cloning NeMo repository...")
    subprocess.run([
        "git", "clone", "https://github.com/prekshivyas/NeMo.git", "/tmp/nemo"
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", "/tmp/nemo"
    ], check=True)

    print("Cloning NeMo RUN repository...")
    subprocess.run([
        "git", "clone", "https://github.com/prekshivyas/Run.git", "/tmp/nemo_run"
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", "/tmp/nemo_run"
    ], check=True)

def generate_configs(args_dict: Dict[str, Any]):
    """Generate AutoTune configurations."""
    # Setup environment first
    setup_nemo_environment()
    
    from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import generate
    from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
    
    # Convert args_dict to AutoTuneArgs
    args = AutoTuneArgs(**args_dict)
    result = generate(**args.to_dict())
    print(f"Generated {len(result)} configurations")

def run_pretraining(config_dir: str, model: str, sequential: bool, run_all: bool):
    """Run AutoTune pretraining."""
    # Setup environment first
    setup_nemo_environment()
    
    from nemo.collections.llm.tools.autotuner.core.pretraining import run_pretraining as run_pretraining_impl
    from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
    
    # Load args from file
    args = AutoTuneArgs.load_from_file(f'{config_dir}/{model}/args.json')
    args.sequential = sequential
    args.metadata['run_all'] = run_all
    
    # Run pretraining
    results = run_pretraining_impl(
        base_config=args.get_base_config(),
        configs=args.metadata.get('configs', {}),
        base_config_matches=args.metadata.get('base_config_matches', []),
        sequential=args.sequential,
        executor_config=args.get_executor_config(),
        memory_analysis=args.get_memory_analysis(),
        run_all=args.metadata.get('run_all', False)
    )
    print(f"Pretraining completed with {len(results)} configurations")

def analyze_results(config_dir: str, model: str, path: str, log_prefix: str, 
                   top_n: int, force_reconstruct: bool, cost_per_gpu_hour: float, quiet: bool):
    """Analyze AutoTune results."""
    # Setup environment first
    setup_nemo_environment()
    
    from nemo.collections.llm.tools.autotuner.core.performance import results
    from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
    import os
    
    # Load args from file
    args_file_path = f'{config_dir}/{model}/args.json'
    if not os.path.exists(args_file_path):
        print(f"ERROR: args.json file not found at {args_file_path}")
        return
    
    args = AutoTuneArgs.load_from_file(args_file_path)
    
    # Analyze results
    results(
        args=args,
        logs_path=path,
        log_prefix=log_prefix,
        top_n=top_n,
        force_reconstruct=force_reconstruct,
        cost_per_gpu_hour=cost_per_gpu_hour,
        quiet=quiet
    )
    print("Results analysis completed!")

def list_configurations(config_dir: str, model: str):
    """List generated configurations."""
    # Setup environment first
    setup_nemo_environment()
    
    from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import list_configs
    
    list_configs(config_dir, model)
    print("Configuration listing completed!")

def launch_generate_remote(args_dict: Dict[str, Any], launcher_node_group: str, training_node_group: str, mount_from: Optional[str] = None):
    """Launch generate step using remote executor."""
    
    mounts=[
    {
        "path": "/",
        "mount_path": "/nemo-workspace",
        "from": "node-nfs:lepton-shared-fs"
    }
]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="python:3.11",
        node_group=launcher_node_group,
        mounts=mounts
    )
    
    # Run on remote executor
    run.run(
        run.Partial(generate_configs, args_dict),
        executor=executor
    )
    
    print("Configuration generation completed!")

def launch_run_remote(config_dir: str, model: str, sequential: bool = False, run_all: bool = False, 
                     launcher_node_group: Optional[str] = None, training_node_group: Optional[str] = None, 
                     mount_from: Optional[str] = None):
    """Launch run step using remote executor."""
    
    mounts=[
    {
        "path": "/",
        "mount_path": "/nemo-workspace",
        "from": "node-nfs:lepton-shared-fs"
    }
]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="gpu.a100",
        container_image="python:3.11",
        node_group=training_node_group,
        nodes=1,
        nprocs_per_node=8,
        mounts=mounts
    )
    
    # Run on remote executor
    run.run(
        run.Partial(run_pretraining, config_dir, model, sequential, run_all),
        executor=executor
    )
    
    print("AutoTune pretraining completed!")

def launch_results_remote(config_dir: str, model: str, path: str, log_prefix: str, 
                        top_n: int = 10, force_reconstruct: bool = False, 
                        cost_per_gpu_hour: float = 24.0, quiet: bool = False,
                        launcher_node_group: Optional[str] = None, 
                        training_node_group: Optional[str] = None, 
                        mount_from: Optional[str] = None):
    """Launch results collection step using remote executor."""
    
    mounts=[
    {
        "path": "/",
        "mount_path": "/nemo-workspace",
        "from": "node-nfs:lepton-shared-fs"
    }
]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="python:3.11",
        node_group=launcher_node_group,
        mounts=mounts
    )
    
    # Run on remote executor
    run.run(
        run.Partial(analyze_results, config_dir, model, path, log_prefix, top_n, force_reconstruct, cost_per_gpu_hour, quiet),
        executor=executor
    )
    
    print("Results analysis completed!")

def launch_list_configs_remote(config_dir: str, model: str, launcher_node_group: Optional[str] = None, 
                              mount_from: Optional[str] = None, training_node_group: Optional[str] = None):
    """Launch list-configs step using remote executor."""
    
    mounts=[
        {
        "path": "/",
        "mount_path": "/nemo-workspace",
        "from": "node-nfs:lepton-shared-fs"
        }
    ]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="python:3.11",
        node_group=launcher_node_group,
        mounts=mounts
    )
    
    # Run on remote executor
    run.run(
        run.Partial(list_configurations, config_dir, model),
        executor=executor
    )
    
    print("Configuration listing completed!")

def list_models():
    """List supported models - runs locally since it's just a lookup."""
    from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import list_models
    list_models()

def create_parser():
    """Create argument parser for direct CLI usage."""
    parser = argparse.ArgumentParser(
        description="AutoTuner Launcher using NeMo Run Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python launcher.py generate --config-dir /path/to/configs --model llama3_70b
                python launcher.py run --config-dir /path/to/configs --model llama3_70b
                python launcher.py results --config-dir /path/to/configs --model llama3_70b --path /path/to/logs
                python launcher.py list-configs --config-dir /path/to/configs --model llama3_70b
                """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate AutoTune configurations')
    generate_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    generate_parser.add_argument('--model', required=True, help='Model name')
    generate_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    generate_parser.add_argument('--training-node-group', required=True, help='Training node group')
    # Add other generate arguments as needed
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run AutoTune pretraining')
    run_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    run_parser.add_argument('--model', required=True, help='Model name')
    run_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    run_parser.add_argument('--training-node-group', required=True, help='Training node group')
    run_parser.add_argument('--sequential', action='store_true', help='Run sequentially')
    run_parser.add_argument('--run-all', action='store_true', help='Run all configurations')
    
    # Results command
    results_parser = subparsers.add_parser('results', help='Analyze AutoTune results')
    results_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    results_parser.add_argument('--model', required=True, help='Model name')
    results_parser.add_argument('--path', required=True, help='Path to logs')
    results_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    results_parser.add_argument('--training-node-group', required=True, help='Training node group')
    results_parser.add_argument('--log-prefix', default='', help='Log prefix')
    results_parser.add_argument('--top-n', type=int, default=5, help='Top N results')
    results_parser.add_argument('--force-reconstruct', action='store_true', help='Force reconstruction')
    results_parser.add_argument('--cost-per-gpu-hour', type=float, default=24.0, help='Cost per GPU hour')
    results_parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    # List-configs command
    list_configs_parser = subparsers.add_parser('list-configs', help='List AutoTune configurations')
    list_configs_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    list_configs_parser.add_argument('--model', required=True, help='Model name')
    list_configs_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    
    return parser

def main():
    """Main entry point for direct CLI usage."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == 'generate':
        # Convert args to dict for generate
        args_dict = vars(args)
        launch_generate_remote(args_dict, args.launcher_node_group, args.training_node_group)
    elif args.command == 'run':
        launch_run_remote(args.config_dir, args.model, args.sequential, args.run_all, 
                         args.launcher_node_group, args.training_node_group)
    elif args.command == 'results':
        launch_results_remote(args.config_dir, args.model, args.path, args.log_prefix,
                            args.top_n, args.force_reconstruct, args.cost_per_gpu_hour, args.quiet,
                            args.launcher_node_group, args.training_node_group)
    elif args.command == 'list-configs':
        launch_list_configs_remote(args.config_dir, args.model, args.launcher_node_group)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
