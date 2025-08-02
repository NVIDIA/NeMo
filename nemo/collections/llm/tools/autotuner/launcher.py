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
    container_image: str = "nvcr.io/nvidia/nemo:25.07",
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
        "PYTHONPATH": "/tmp/nemo:/tmp/nemo_run:$PYTHONPATH",
        "TORCH_HOME": "/workspace/.cache",
    }
    
    # Map AUTOTUNER_* environment variables to LEPTON_* variables for remote container
    autotuner_env_vars = [
        "AUTOTUNER_WORKSPACE_ID",
        "AUTOTUNER_WORKSPACE_URL", 
        "AUTOTUNER_TOKEN"
    ]
    
    # Copy AUTOTUNER_* variables to LEPTON_AUTOTUNER_* variables if they exist
    for var in autotuner_env_vars:
        if var in os.environ:
            lepton_var = var.replace("AUTOTUNER_", "LEPTON_AUTOTUNER_")
            default_env_vars[lepton_var] = os.environ[var]
    
    env_vars.update(default_env_vars)
    
    return run.LeptonExecutor(
        resource_shape=resource_shape,
        container_image=container_image,
        nemo_run_dir=nemo_run_dir,
        mounts=mounts,
        node_group=node_group,
        nodes=nodes,
        shared_memory_size=1024,
    )

def create_autotune_script(script_type: str, **kwargs) -> run.Script:
    """Create a script for autotune operations."""
    script_content = f"""#!/usr/bin/env python3
import sys
import os
import subprocess

# Add /tmp/nemo to Python path as first thing
sys.path.insert(0, "/tmp/nemo")

# Print Python path for debugging
print("Python path:", sys.path)

# Set Lepton environment variables from AUTOTUNER variables
if "LEPTON_AUTOTUNER_WORKSPACE_ID" in os.environ:
    os.environ["LEPTON_WORKSPACE_ID"] = os.environ["LEPTON_AUTOTUNER_WORKSPACE_ID"]
if "LEPTON_AUTOTUNER_WORKSPACE_URL" in os.environ:
    os.environ["LEPTON_WORKSPACE_URL"] = os.environ["LEPTON_AUTOTUNER_WORKSPACE_URL"]
if "LEPTON_AUTOTUNER_TOKEN" in os.environ:
    os.environ["LEPTON_TOKEN"] = os.environ["LEPTON_AUTOTUNER_TOKEN"]

def setup_nemo_environment():
    # Install NeMo directly from GitHub
    print("Installing NeMo from GitHub...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/prekshivyas/NeMo.git"
    ], check=True)

if __name__ == "__main__":
    # Setup environment first
    setup_nemo_environment()
    
    # Print Python path again after setup
    print("Python path after setup:", sys.path)
    
    # Now import the modules after environment is set up
    from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import generate, list_configs
    from nemo.collections.llm.tools.autotuner.core.pretraining import run_pretraining as run_pretraining_impl
    from nemo.collections.llm.tools.autotuner.core.performance import results
    from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
    import json
    
    # Get arguments from environment
    script_type = "{script_type}"
"""

    if script_type == "generate":
        script_content += f"""
    # Generate configs
    args_dict = {kwargs.get('args_dict', {})}
    args = AutoTuneArgs(**args_dict)
    result = generate(**args.to_dict())
"""
    elif script_type == "run":
        script_content += f"""
    # Run pretraining
    config_dir = "{kwargs.get('config_dir', '')}"
    model = "{kwargs.get('model', '')}"
    sequential = {kwargs.get('sequential', False)}
    run_all = {kwargs.get('run_all', False)}
    
    args = AutoTuneArgs.load_from_file(f'{{config_dir}}/{{model}}/args.json')
    args.sequential = sequential
    args.metadata['run_all'] = run_all
    
    # Add environment variables to executor config
    executor_config = args.get_executor_config()
    executor_config['env_vars'] = {{
        'LEPTON_WORKSPACE_ID': os.environ.get('LEPTON_AUTOTUNER_WORKSPACE_ID', ''),
        'LEPTON_WORKSPACE_URL': os.environ.get('LEPTON_AUTOTUNER_WORKSPACE_URL', ''),
        'LEPTON_TOKEN': os.environ.get('LEPTON_AUTOTUNER_TOKEN', '')
    }}
    
    results = run_pretraining_impl(
        base_config=args.get_base_config(),
        configs=args.metadata.get('configs', {{}}),
        base_config_matches=args.metadata.get('base_config_matches', []),
        sequential=args.sequential,
        executor_config=executor_config,
        memory_analysis=args.get_memory_analysis(),
        run_all=args.metadata.get('run_all', False)
    )
"""
    elif script_type == "results":
        script_content += f"""
    # Analyze results
    config_dir = "{kwargs.get('config_dir', '')}"
    model = "{kwargs.get('model', '')}"
    path = "{kwargs.get('path', '')}"
    log_prefix = "{kwargs.get('log_prefix', '')}"
    top_n = {kwargs.get('top_n', 10)}
    force_reconstruct = {kwargs.get('force_reconstruct', False)}
    cost_per_gpu_hour = {kwargs.get('cost_per_gpu_hour', 24.0)}
    quiet = {kwargs.get('quiet', False)}
    
    args_file_path = f'{{config_dir}}/{{model}}/args.json'
    if not os.path.exists(args_file_path):
        print(f"ERROR: args.json file not found at {{args_file_path}}")
        exit(1)
    
    args = AutoTuneArgs.load_from_file(args_file_path)
    
    results(
        args=args,
        logs_path=path,
        log_prefix=log_prefix,
        top_n=top_n,
        force_reconstruct=force_reconstruct,
        cost_per_gpu_hour=cost_per_gpu_hour,
        quiet=quiet
    )
"""
    elif script_type == "list_configs":
        script_content += f"""
    # List configurations
    config_dir = "{kwargs.get('config_dir', '')}"
    model = "{kwargs.get('model', '')}"
    
    list_configs(config_dir, model)
"""

    return run.Script(inline=script_content, entrypoint="python")

def launch_generate_remote(args_dict: Dict[str, Any], launcher_node_group: str, training_node_group: str, mount_from: Optional[str] = None):
    """Launch generate step using remote executor."""
    
    mounts=[
    {
        "path": "/",
        "mount_path": "/nemo-workspace",
        "from": "node-nfs:az-files-nfs-vol"
    }
]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="nvcr.io/nvidia/nemo:25.07",
        node_group=launcher_node_group,
        mounts=mounts
    )
    
    # Create and run script
    script = create_autotune_script("generate", args_dict=args_dict)
    run.run(
        script,
        executor=executor,
        name="autotune-generate"
    )

def launch_run_remote(config_dir: str, model: str, sequential: bool = False, run_all: bool = False, 
                     launcher_node_group: Optional[str] = None, training_node_group: Optional[str] = None, 
                     mount_from: Optional[str] = None):
    """Launch run step using remote executor."""
    
    # Map AUTOTUNER_* environment variables to LEPTON_* variables for local NeMo Run process
    autotuner_env_vars = [
        "AUTOTUNER_WORKSPACE_ID",
        "AUTOTUNER_WORKSPACE_URL", 
        "AUTOTUNER_TOKEN"
    ]
    
    # Set LEPTON_* variables in local environment for NeMo Run process
    for var in autotuner_env_vars:
        if var in os.environ:
            lepton_var = var.replace("AUTOTUNER_", "LEPTON_")
            os.environ[lepton_var] = os.environ[var]
    
    mounts=[
    {
        "path": "/",
        "mount_path": "/nemo-workspace",
        "from": "node-nfs:az-files-nfs-vol"
    }
]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="nvcr.io/nvidia/nemo:25.07",
        node_group=training_node_group,
        nodes=1,
        mounts=mounts
    )
    
    # Create and run script
    script = create_autotune_script("run", config_dir=config_dir, model=model, sequential=sequential, run_all=run_all)
    run.run(
        script,
        executor=executor,
        name="autotune-run"
    )

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
        "from": "node-nfs:az-files-nfs-vol"
    }
]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="nvcr.io/nvidia/nemo:25.07",
        node_group=launcher_node_group,
        mounts=mounts
    )
    
    # Create and run script
    script = create_autotune_script("results", config_dir=config_dir, model=model, path=path, 
                                   log_prefix=log_prefix, top_n=top_n, force_reconstruct=force_reconstruct, 
                                   cost_per_gpu_hour=cost_per_gpu_hour, quiet=quiet)
    run.run(
        script,
        executor=executor,
        name="autotune-results"
    )

def launch_list_configs_remote(config_dir: str, model: str, launcher_node_group: Optional[str] = None, 
                              mount_from: Optional[str] = None, training_node_group: Optional[str] = None):
    """Launch list-configs step using remote executor."""
    
    mounts=[
        {
        "path": "/",
        "mount_path": "/nemo-workspace",
        "from": "nnode-nfs:az-files-nfs-vol"
        }
    ]
    
    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="nvcr.io/nvidia/nemo:25.07",
        node_group=launcher_node_group,
        mounts=mounts
    )
    
    # Create and run script
    script = create_autotune_script("list_configs", config_dir=config_dir, model=model)
    run.run(
        script,
        executor=executor,
        name="autotune-list-configs"
    )

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
