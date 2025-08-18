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

"""
AutoTuner Launcher - Using NeMo Run Experiment approach
Uses run.Experiment with run.LeptonExecutor and run.Partial/run.Script for clean execution.
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Optional

import nemo_run as run

logger = logging.getLogger(__name__)


def parse_comma_separated_values(value: str) -> List[int]:
    """Parse comma-separated string values into a list of integers."""
    if not value or value.lower() == 'none':
        return []
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_virtual_pipeline_values(value: str) -> Optional[List[int]]:
    """Parse virtual pipeline values, handling 'None' specially."""
    if not value or value.lower() == 'none':
        return None
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def create_lepton_executor(
    resource_shape: str = "cpu.small",
    container_image: str = "nvcr.io/nvidia/nemo:25.07",
    nemo_run_dir: str = "/nemo-workspace/nemo-run",
    mounts: Optional[list] = None,
    node_group: Optional[str] = None,
    nodes: int = 1,
    env_vars: Optional[Dict[str, str]] = None,
) -> run.LeptonExecutor:
    """Create a LeptonExecutor with the specified configuration."""

    if mounts is None:
        mounts = []

    if env_vars is None:
        env_vars = {}

    # Add default environment variables
    default_env_vars = {
        "TORCH_HOME": "/workspace/.cache",
    }

    # Map AUTOTUNER_* environment variables to LEPTON_* variables for remote container
    autotuner_env_vars = ["AUTOTUNER_WORKSPACE_ID", "AUTOTUNER_WORKSPACE_URL", "AUTOTUNER_TOKEN"]

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
    lepton_workspace_id = kwargs.get('lepton_workspace_id')
    lepton_token = kwargs.get('lepton_token')

    script_content = f"""#!/usr/bin/env python3
import sys
import os
import subprocess

def setup_nemo_environment():
    # First uninstall existing NeMo framework
    print("Uninstalling existing NeMo framework...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "nemo-toolkit"
        ], check=True, capture_output=False)
        print("Successfully uninstalled existing NeMo framework")
    except subprocess.CalledProcessError:
        print("No existing NeMo framework found or already uninstalled")
    
    # Install NeMo directly from GitHub repository
    print("Installing NeMo from GitHub repository...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "git+https://github.com/prekshivyas/NeMo.git"
    ], check=True, capture_output=False)
    print("Successfully installed NeMo from GitHub repository")
    
    print("Authenticating with Lepton CLI...")
    subprocess.run([
        "lep", "login", "-c", "{lepton_workspace_id}:{lepton_token}"
    ], check=True, capture_output=False)
    print("Successfully authenticated with Lepton workspace: {lepton_workspace_id}")

if __name__ == "__main__":
    setup_nemo_environment()
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
    logs_path = "{kwargs.get('logs_path', '')}"
    log_prefix = "{kwargs.get('log_prefix', '')}"
    top_n = {kwargs.get('top_n', 10)}
    cost_per_gpu_hour = {kwargs.get('cost_per_gpu_hour', 3.0)}
    
    args_file_path = f'{{config_dir}}/{{model}}/args.json'
    if not os.path.exists(args_file_path):
        print(f"ERROR: args.json file not found at {{args_file_path}}")
        exit(1)
    
    args = AutoTuneArgs.load_from_file(args_file_path)
    
    results(
        args=args,
        logs_path=logs_path,
        log_prefix=log_prefix,
        top_n=top_n,
        cost_per_gpu_hour=cost_per_gpu_hour
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


def launch_generate_remote(
    args_dict: Dict[str, Any],
    launcher_node_group: str,
    training_node_group: str,
    mount_from: str,
    mount_source_path: str,
    mount_path: str,
    resource_shape: str = "gpu.8xh200",
    container_image: str = "nvcr.io/nvidia/nemo:25.04",
    nodes: int = 8,
    gpus_per_node: int = 8,
    seq_length: int = 8192,
    num_tokens_in_b: int = 1000,
    global_batch_sizes: str = "256",
    tensor_parallel_sizes: str = "2",
    pipeline_parallel_sizes: str = "2",
    virtual_pipeline_model_parallel_sizes: Optional[str] = "None",
    expert_parallel_sizes: Optional[str] = "1",
    max_model_parallel_size: int = 64,
    context_parallel_sizes: str = "1",
    micro_batch_sizes: str = "1",
    max_steps_per_run: int = 50,
    max_steps: int = 50,
    logs_subdir: str = "/nemo-workspace/autotuner/new/logs",
):
    """Launch generate step using remote executor."""

    # Convert string arguments to proper types and update args_dict
    args_dict.update(
        {
            'nodes': int(nodes),
            'gpus_per_node': int(gpus_per_node),
            'resource_shape': resource_shape,
            'seq_length': int(seq_length),
            'num_tokens_in_b': int(num_tokens_in_b),
            'global_batch_sizes': parse_comma_separated_values(global_batch_sizes),
            'tensor_parallel_sizes': parse_comma_separated_values(tensor_parallel_sizes),
            'pipeline_parallel_sizes': parse_comma_separated_values(pipeline_parallel_sizes),
            'virtual_pipeline_model_parallel_sizes': parse_virtual_pipeline_values(
                virtual_pipeline_model_parallel_sizes
            ),
            'expert_parallel_sizes': parse_comma_separated_values(expert_parallel_sizes),
            'max_model_parallel_size': int(max_model_parallel_size),
            'context_parallel_sizes': parse_comma_separated_values(context_parallel_sizes),
            'micro_batch_sizes': parse_comma_separated_values(micro_batch_sizes),
            'max_steps_per_run': int(max_steps_per_run),
            'max_steps': int(max_steps),
            'logs_subdir': logs_subdir,
        }
    )

    mounts = [{"path": mount_source_path, "mount_path": mount_path, "from": mount_from}]

    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image=container_image,
        node_group=launcher_node_group,
        mounts=mounts,
    )

    # Create and run script
    script = create_autotune_script(
        "generate",
        args_dict=args_dict,
        lepton_workspace_id=os.environ.get('LEPTON_AUTOTUNER_WORKSPACE_ID', ''),
        lepton_token=os.environ.get('LEPTON_AUTOTUNER_TOKEN', ''),
    )
    run.run(script, executor=executor, name="autotune-generate")


def launch_run_remote(
    config_dir: str,
    model: str,
    mount_from: str,
    mount_source_path: str,
    mount_path: str,
    launcher_node_group: str,
    training_node_group: str,
    sequential: bool = False,
    run_all: bool = False,
):
    """Launch run step using remote executor."""

    # Map AUTOTUNER_* environment variables to LEPTON_* variables for local NeMo Run process
    autotuner_env_vars = ["AUTOTUNER_WORKSPACE_ID", "AUTOTUNER_WORKSPACE_URL", "AUTOTUNER_TOKEN"]

    # Set LEPTON_* variables in local environment for NeMo Run process
    for var in autotuner_env_vars:
        if var in os.environ:
            lepton_var = var.replace("AUTOTUNER_", "LEPTON_")
            os.environ[lepton_var] = os.environ[var]

    mounts = [{"path": mount_source_path, "mount_path": mount_path, "from": mount_from}]

    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="nvcr.io/nvidia/nemo:25.07",
        node_group=training_node_group,
        nodes=1,
        mounts=mounts,
    )

    # Create and run script
    script = create_autotune_script(
        "run",
        config_dir=config_dir,
        model=model,
        sequential=sequential,
        run_all=run_all,
        lepton_workspace_id=os.environ.get('LEPTON_AUTOTUNER_WORKSPACE_ID', ''),
        lepton_token=os.environ.get('LEPTON_AUTOTUNER_TOKEN', ''),
    )
    run.run(script, executor=executor, name="autotune-run")


def launch_results_remote(
    config_dir: str,
    model: str,
    logs_path: str,
    log_prefix: str,
    mount_from: str,
    mount_source_path: str,
    mount_path: str,
    launcher_node_group: str,
    top_n: int = 10,
    cost_per_gpu_hour: float = 3.0,
):
    """Launch results collection step using remote executor."""
    mounts = [{"path": mount_source_path, "mount_path": mount_path, "from": mount_from}]

    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="nvcr.io/nvidia/nemo:25.07",
        node_group=launcher_node_group,
        mounts=mounts,
    )

    # Create and run script
    script = create_autotune_script(
        "results",
        config_dir=config_dir,
        model=model,
        logs_path=logs_path,
        log_prefix=log_prefix,
        top_n=top_n,
        cost_per_gpu_hour=cost_per_gpu_hour,
        lepton_workspace_id=os.environ.get('LEPTON_AUTOTUNER_WORKSPACE_ID', ''),
        lepton_token=os.environ.get('LEPTON_AUTOTUNER_TOKEN', ''),
    )
    run.run(script, executor=executor, name="autotune-results")


def launch_list_configs_remote(
    config_dir: str, model: str, mount_from: str, mount_source_path: str, mount_path: str, launcher_node_group: str
):
    """Launch list-configs step using remote executor."""
    mounts = [{"path": mount_source_path, "mount_path": mount_path, "from": mount_from}]

    # Create executor for remote execution
    executor = create_lepton_executor(
        resource_shape="cpu.small",
        container_image="nvcr.io/nvidia/nemo:25.07",
        node_group=launcher_node_group,
        mounts=mounts,
    )

    # Create and run script
    script = create_autotune_script(
        "list_configs",
        config_dir=config_dir,
        model=model,
        lepton_workspace_id=os.environ.get('LEPTON_AUTOTUNER_WORKSPACE_ID', ''),
        lepton_token=os.environ.get('LEPTON_AUTOTUNER_TOKEN', ''),
    )
    run.run(script, executor=executor, name="autotune-list-configs")


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
                python launcher.py generate --config-dir /path/to/configs --model llama3_70b --launcher-node-group group1 --training-node-group group2 --mount-from node-nfs:shared --mount-source-path /local/path --mount-path /nemo-workspace
                python launcher.py run --config-dir /path/to/configs --model llama3_70b --launcher-node-group group1 --training-node-group group2 --mount-from node-nfs:shared --mount-source-path /local/path --mount-path /nemo-workspace
                python launcher.py results --config-dir /path/to/configs --model llama3_70b --logs-path /path/to/logs --launcher-node-group group1 --mount-from node-nfs:shared --mount-source-path /local/path --mount-path /nemo-workspace
                python launcher.py list-configs --config-dir /path/to/configs --model llama3_70b --launcher-node-group group1 --mount-from node-nfs:shared --mount-source-path /local/path --mount-path /nemo-workspace
                """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate AutoTune configurations')
    generate_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    generate_parser.add_argument('--model', required=True, help='Model name')
    generate_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    generate_parser.add_argument('--training-node-group', required=True, help='Training node group')
    generate_parser.add_argument('--mount-from', required=True, help='Mount source')
    generate_parser.add_argument('--mount-source-path', required=True, help='Mount source path')
    generate_parser.add_argument('--mount-path', required=True, help='Mount destination path')
    generate_parser.add_argument('--resource-shape', default='gpu.8xh200', help='Resource shape (e.g., gpu.8xh200)')
    generate_parser.add_argument('--container-image', default='nvcr.io/nvidia/nemo:25.07', help='Container image')
    generate_parser.add_argument('--nodes', type=int, default=8, help='Number of nodes')
    generate_parser.add_argument('--gpus-per-node', type=int, default=8, help='GPUs per node')
    generate_parser.add_argument('--seq-length', type=int, default=8192, help='Sequence length')
    generate_parser.add_argument('--num-tokens-in-b', type=int, default=1000, help='Number of tokens in billions')
    generate_parser.add_argument('--global-batch-sizes', default='256', help='Global batch sizes (comma-separated)')
    generate_parser.add_argument(
        '--tensor-parallel-sizes', default='2', help='Tensor parallel sizes (comma-separated)'
    )
    generate_parser.add_argument(
        '--pipeline-parallel-sizes', default='2', help='Pipeline parallel sizes (comma-separated)'
    )
    generate_parser.add_argument(
        '--virtual-pipeline-model-parallel-sizes',
        default='None',
        help='Virtual pipeline parallel sizes (comma-separated or None)',
    )
    generate_parser.add_argument('--max-model-parallel-size', type=int, default=64, help='Maximum model parallel size')
    generate_parser.add_argument(
        '--context-parallel-sizes', default='1', help='Context parallel sizes (comma-separated)'
    )
    generate_parser.add_argument(
        '--expert-parallel-sizes', default='1', help='Expert parallel sizes (comma-separated)'
    )
    generate_parser.add_argument('--micro-batch-sizes', default='1', help='Micro batch sizes (comma-separated)')
    generate_parser.add_argument('--max-steps-per-run', type=int, default=50, help='Maximum steps per run')
    generate_parser.add_argument('--max-steps', type=int, default=50, help='Maximum steps')
    generate_parser.add_argument(
        '--logs-subdir', default='/nemo-workspace/autotuner/new/logs', help='Logs subdirectory'
    )

    # Run command
    run_parser = subparsers.add_parser('run', help='Run AutoTune pretraining')
    run_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    run_parser.add_argument('--model', required=True, help='Model name')
    run_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    run_parser.add_argument('--training-node-group', required=True, help='Training node group')
    run_parser.add_argument('--sequential', action='store_true', help='Run sequentially')
    run_parser.add_argument('--run-all', action='store_true', help='Run all configurations')
    run_parser.add_argument('--mount-from', required=True, help='Mount source')
    run_parser.add_argument('--mount-source-path', required=True, help='Mount source path')
    run_parser.add_argument('--mount-path', required=True, help='Mount destination path')

    # Results command
    results_parser = subparsers.add_parser('results', help='Analyze AutoTune results')
    results_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    results_parser.add_argument('--model', required=True, help='Model name')
    results_parser.add_argument('--logs-path', required=True, help='Path to logs')
    results_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    results_parser.add_argument('--log-prefix', default='', help='Log prefix')
    results_parser.add_argument('--top-n', type=int, default=5, help='Top N results')
    results_parser.add_argument('--cost-per-gpu-hour', type=float, default=3.0, help='Cost per GPU hour')
    results_parser.add_argument('--mount-from', required=True, help='Mount source')
    results_parser.add_argument('--mount-path', required=True, help='Mount destination path')
    results_parser.add_argument('--mount-source-path', required=True, help='Mount source path')

    # List-configs command
    list_configs_parser = subparsers.add_parser('list-configs', help='List AutoTune configurations')
    list_configs_parser.add_argument('--config-dir', required=True, help='Configuration directory')
    list_configs_parser.add_argument('--model', required=True, help='Model name')
    list_configs_parser.add_argument('--launcher-node-group', required=True, help='Launcher node group')
    list_configs_parser.add_argument('--mount-from', required=True, help='Mount source')
    list_configs_parser.add_argument('--mount-path', required=True, help='Mount destination path')
    list_configs_parser.add_argument('--mount-source-path', required=True, help='Mount source path')

    return parser


def main():
    """Main entry point for direct CLI usage."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate required arguments
    if not args.command:
        parser.print_help()
        return

    mount_params = ['mount_from', 'mount_source_path', 'mount_path']
    for param in mount_params:
        if hasattr(args, param) and getattr(args, param) is not None:
            if not getattr(args, param).strip():
                logger.error(f"Error: {param} cannot be empty")
                return

    try:
        if args.command == 'generate':
            # Convert args to dict for generate
            args_dict = vars(args)
            launch_generate_remote(
                args_dict,
                args.launcher_node_group,
                args.training_node_group,
                args.mount_from,
                args.mount_source_path,
                args.mount_path,
                resource_shape=args.resource_shape,
                container_image=args.container_image,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                seq_length=args.seq_length,
                num_tokens_in_b=args.num_tokens_in_b,
                global_batch_sizes=args.global_batch_sizes,
                tensor_parallel_sizes=args.tensor_parallel_sizes,
                pipeline_parallel_sizes=args.pipeline_parallel_sizes,
                virtual_pipeline_model_parallel_sizes=args.virtual_pipeline_model_parallel_sizes,
                expert_parallel_sizes=args.expert_parallel_sizes,
                max_model_parallel_size=args.max_model_parallel_size,
                context_parallel_sizes=args.context_parallel_sizes,
                micro_batch_sizes=args.micro_batch_sizes,
                max_steps_per_run=args.max_steps_per_run,
                max_steps=args.max_steps,
                logs_subdir=args.logs_subdir,
            )
        elif args.command == 'run':
            launch_run_remote(
                args.config_dir,
                args.model,
                args.mount_from,
                args.mount_source_path,
                args.mount_path,
                args.launcher_node_group,
                args.training_node_group,
                args.sequential,
                args.run_all,
            )
        elif args.command == 'results':
            launch_results_remote(
                args.config_dir,
                args.model,
                args.logs_path,
                args.log_prefix,
                args.mount_from,
                args.mount_source_path,
                args.mount_path,
                args.launcher_node_group,
                args.top_n,
                args.cost_per_gpu_hour,
            )
        elif args.command == 'list-configs':
            launch_list_configs_remote(
                args.config_dir,
                args.model,
                args.mount_from,
                args.mount_source_path,
                args.mount_path,
                args.launcher_node_group,
            )
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        raise


if __name__ == "__main__":
    main()
