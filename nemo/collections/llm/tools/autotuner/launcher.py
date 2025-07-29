#!/usr/bin/env python3
"""
Data Mover Launcher - Remote execution layer for autotuner
This layer handles remote job execution and calls the actual autotuner functions.
Can be used both as a direct CLI tool and as a module imported by the CLI layer.
"""

import argparse
import os
import subprocess
import sys
import time
import threading
import json
from pathlib import Path
from leptonai.api.v2.client import APIClient
from leptonai.api.v1.types.job import LeptonJob, LeptonJobState, LeptonJobUserSpec
from leptonai.api.v1.types.deployment import LeptonContainer, Mount, EnvVar
from leptonai.api.v1.types.affinity import LeptonResourceAffinity
from leptonai.api.v1.types.common import Metadata, LeptonVisibility
from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import list_models
from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs


def list_available_node_groups():
    """List all available node groups for debugging."""
    try:
        client = APIClient()
        node_groups = client.nodegroup.list_all()
        print("Available node groups:")
        for ng in node_groups:
            print(f"  - {ng.metadata.name} (ID: {ng.metadata.id_})")
        return [ng.metadata.name for ng in node_groups]
    except Exception as e:
        print(f"Error listing node groups: {e}")
        return []

def create_job(step_name, command, resource_shape="cpu.small", launcher_node_group=None, training_node_group=None, mount_from=None):
    """Create and launch a job directly using Lepton API."""
    print(f"Launching {step_name} step...")
    client = APIClient()

    # Install local NeMo changes once using a flag file
    nemo_setup = "test -f /nemo-workspace/.nemo_installed || (echo 'Checking NeMo installation...' && ls -la /nemo-workspace/ && if [ -d '/nemo-workspace/nemo-run/code/NeMo' ]; then echo 'Installing local NeMo changes...' && pip install -e /nemo-workspace/nemo-run/code/NeMo && touch /nemo-workspace/.nemo_installed && echo 'NeMo installation completed'; else echo 'NeMo directory not found, skipping installation'; fi)"
    cmd = f"{nemo_setup} && {command}"
    
    envs = [
        EnvVar(name="PYTHONPATH", value="/nemo-workspace/nemo-run/code/NeMo:$PYTHONPATH"),
        EnvVar(name="TORCH_HOME", value="/workspace/.cache"),
    ]
    
    # Configure node group affinity if specified
    affinity = None
    if launcher_node_group:
        try:
            print(f"Configuring launcher node group: {launcher_node_group}")
            # Get available node groups
            node_groups = client.nodegroup.list_all()
            print(f"Available node groups: {[ng.metadata.name for ng in node_groups]}")
            
            # Create mapping of name to node group
            node_group_map = {ng.metadata.name: ng for ng in node_groups}
            
            if launcher_node_group in node_group_map:
                node_group_id = node_group_map[launcher_node_group]
                affinity = LeptonResourceAffinity(
                    allowed_dedicated_node_groups=[node_group_id.metadata.id_]
                )
                print(f"✓ Using launcher node group: {launcher_node_group} (ID: {node_group_id.metadata.id_})")
            else:
                print(f"⚠ Warning: Launcher node group '{launcher_node_group}' not found.")
                print(f"Available groups: {list(node_group_map.keys())}")
                print("Proceeding without node group specification...")
        except Exception as e:
            print(f"⚠ Warning: Could not configure launcher node group '{launcher_node_group}': {e}")
            print("Proceeding without node group specification...")
    else:
        print("No launcher node group specified, using default scheduling...")
    
    # Use the mount_from parameter or default to lepton-shared-fs
    mount_path = mount_from if mount_from else "az-files-nfs-vol"
    print(f"Using mount: {mount_path}")
    
    job_spec = LeptonJobUserSpec(
        resource_shape=resource_shape,
        container=LeptonContainer(
            image="nvcr.io/nvidia/nemo:25.04",
            command=["/bin/bash", "-c", cmd]
        ),
        completions=1,
        parallelism=1,
        envs=envs,
        mounts=[
            {
                "path": "/",
                "mount_path": "/nemo-workspace",
                "from": mount_path
            }
        ],
        intra_job_communication=False,
        privileged=False,
        affinity=affinity
    )
    
    job_name = f"autotune-{step_name}-{int(time.time())}"
    job = LeptonJob(
        metadata=Metadata(
            id=job_name,
            name=job_name,
            visibility=LeptonVisibility("private")
        ),
        spec=job_spec
    )
    
    created_job = client.job.create(job)
    job_id = created_job.metadata.id_
    
    print(f"{step_name} step launched with job ID: {job_id}")
    
    while True:
        current_job = client.job.get(job_id)
        status = current_job.status.state
        
        if status in [LeptonJobState.Completed, LeptonJobState.Failed]:
            break
            
        print(f"Job status: {status}")
        time.sleep(5)
    
    if status == LeptonJobState.Completed:
        print(f"{step_name} step completed successfully!")
        print(f"Job ID: {job_id}")
    else:
        print(f"{step_name} step failed with status: {status}")
        print(f"Job ID: {job_id}")
        print("Retrieving job logs for debugging...")
        try:
            replicas = client.job.get_replicas(job_id)
            if replicas:
                logs = client.job.get_log(id_or_job=job_id, replica=replicas[0])
                print("=" * 50)
                print("JOB LOGS:")
                print("=" * 50)
                for line in logs:
                    print(line)
                print("=" * 50)
        except Exception as e:
            print(f"Could not retrieve logs: {e}")
            print(f"You can manually check logs with: lepton job logs {job_id}")
    
    return job_id, status

def launch_generate_remote(args_dict):
    """Launch generate step remotely"""
    command = f"""python3 -c "
                from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import generate
                from nemo.collections.llm.tools.autotuner import AutoTuneArgs

                args = AutoTuneArgs(**{args_dict})

                result = generate(**args.to_dict())
            " """
    launcher_node_group = args_dict.get('launcher_node_group')
    training_node_group = args_dict.get('training_node_group')
    mount_from = args_dict.get('mount_from')
    return create_job("generate", command, resource_shape="cpu.small", launcher_node_group=launcher_node_group, training_node_group=training_node_group, mount_from=mount_from)

def launch_run_remote(config_dir, model, sequential=False, run_all=False, launcher_node_group=None, training_node_group=None, mount_from=None):
    """Launch run step remotely - called by CLI layer."""
    command = f"""python3 -c "
                    from nemo.collections.llm.tools.autotuner.core.pretraining import run_pretraining
                    from nemo.collections.llm.tools.autotuner import AutoTuneArgs

                    args = AutoTuneArgs.load_from_file('{config_dir}/{model}/args.json')
                    args.sequential = {sequential}
                    args.metadata['run_all'] = {run_all}

                    results = run_pretraining(
                        base_config=args.get_base_config(),
                        configs=args.metadata.get('configs', {{}}),
                        base_config_matches=args.metadata.get('base_config_matches', []),
                        sequential=args.sequential,
                        executor_config=args.get_executor_config(),
                        memory_analysis=args.get_memory_analysis(),
                        run_all=args.metadata.get('run_all', False)
                    )
                " """
    return create_job("run", command, resource_shape="cpu.small", launcher_node_group=launcher_node_group, training_node_group=training_node_group, mount_from=mount_from)

def launch_results_remote(config_dir, model, path, log_prefix, top_n=10, force_reconstruct=False, cost_per_gpu_hour=24.0, quiet=False, launcher_node_group=None, training_node_group=None, mount_from=None):
    """Launch results collection step remotely."""
    args = AutoTuneArgs.load_from_file(os.path.join(config_dir, model, "args.json"))
    args.save_to_file(os.path.join(config_dir, model, "args.json"))
    
    command = f"""python3 -c "
                from nemo.collections.llm.tools.autotuner.core.performance import results
                from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
                import os

                args = AutoTuneArgs.load_from_file('/nemo-workspace/{config_dir}/{model}/args.json')
                results(
                    args=args,
                    logs_path='{path}',
                    log_prefix='{log_prefix}',
                    top_n={top_n},
                    force_reconstruct={force_reconstruct},
                    cost_per_gpu_hour={cost_per_gpu_hour},
                    quiet={quiet}
                )
                " """
    
    return create_job("results", command, resource_shape="cpu.small", launcher_node_group=launcher_node_group, training_node_group=training_node_group, mount_from=mount_from)

def launch_list_configs_remote(config_dir, model, launcher_node_group=None, training_node_group=None, mount_from=None):
    """Launch list-configs step remotely - called by CLI layer."""
    command = f"""python3 -c "
                from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import list_configs
                list_configs('{config_dir}', '{model}')
                " """
    return create_job("list-configs", command, resource_shape="cpu.small", launcher_node_group=launcher_node_group, training_node_group=training_node_group, mount_from=mount_from)

def list_models():
    """List supported models - runs locally since it's just a lookup."""
    from nemo.collections.llm.tools.autotuner.core.predictive_config_builder import list_models
    list_models()

def create_parser():
    """Create argument parser for direct CLI usage."""
    parser = argparse.ArgumentParser(
        description="Data Mover Launcher for AutoTune - Direct CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python data_mover_launcher.py generate --model llama2_7b --nodes 2 --gpus-per-node 8 --config-dir ./configs --mount-path /workspace --mount-from node-nfs:shared --node-group my-group --logs-subdir logs
                python data_mover_launcher.py run --config-dir ./configs --model llama2_7b
                python data_mover_launcher.py results --config-dir ./configs --model llama2_7b --path ./logs --log-prefix nemo
                python data_mover_launcher.py list-configs --config-dir ./configs --model llama2_7b
                python data_mover_launcher.py list-models

                """
    )
    
    parser.add_argument('--list-node-groups', action='store_true',
                       help='List available node groups and exit')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate AutoTune configurations')
    generate_parser.add_argument('--model', required=True, help='Model to pretrain')
    generate_parser.add_argument('--nodes', type=int, required=True, help='Number of nodes')
    generate_parser.add_argument('--gpus-per-node', type=int, required=True, help='GPUs per node')
    generate_parser.add_argument('--config-dir', required=True, help='Directory to save configurations')
    generate_parser.add_argument('--mount-path', required=True, help='Mount path in container')
    generate_parser.add_argument('--mount-from', required=True, help='Mount source')
    generate_parser.add_argument('--launcher-node-group', required=True, help='Node group for launcher jobs (CPU jobs)')
    generate_parser.add_argument('--training-node-group', required=True, help='Node group for training jobs (GPU jobs)')
    generate_parser.add_argument('--logs-subdir', required=True, help='Logs subdirectory')
    generate_parser.add_argument('--resource-shape', default='gpu.8xh200', help='GPU resource shape')
    generate_parser.add_argument('--tensor-parallel-sizes', default='1,2', help='Tensor parallel sizes (comma-separated)')
    generate_parser.add_argument('--pipeline-parallel-sizes', default='1,2', help='Pipeline parallel sizes (comma-separated)')
    generate_parser.add_argument('--context-parallel-sizes', default='1,2', help='Context parallel sizes (comma-separated)')
    generate_parser.add_argument('--expert-parallel-sizes', default='1', help='Expert parallel sizes (comma-separated)')
    generate_parser.add_argument('--virtual-pipeline-parallel-sizes', default='1,2', help='Virtual pipeline parallel sizes (comma-separated)')
    generate_parser.add_argument('--global-batch-sizes', default='512', help='Global batch sizes (comma-separated)')
    generate_parser.add_argument('--micro-batch-sizes', default='1,2,4', help='Micro batch sizes (comma-separated)')
    generate_parser.add_argument('--max-steps-per-run', type=int, default=10, help='Maximum steps per run')
    generate_parser.add_argument('--seq-length', type=int, default=8192, help='Sequence length')
    generate_parser.add_argument('--num-tokens-in-b', type=int, default=15000, help='Number of tokens in billions')
    generate_parser.add_argument('--container-image', default='nvcr.io/nvidia/nemo:25.04', help='Container image')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run AutoTune experiments')
    run_parser.add_argument('--config-dir', required=True, help='Directory with generated configurations')
    run_parser.add_argument('--model', required=True, help='Model name')
    run_parser.add_argument('--launcher-node-group', required=True, help='Node group for launcher jobs (CPU jobs)')
    run_parser.add_argument('--training-node-group', required=True, help='Node group for training jobs (GPU jobs)')
    run_parser.add_argument('--sequential', action='store_true', help='Run configurations sequentially')
    run_parser.add_argument('--run-all', action='store_true', help='Run all configurations including OOM risk ones')
    
    # Results command
    results_parser = subparsers.add_parser('results', help='Analyze AutoTune results')
    results_parser.add_argument('--config-dir', required=True, help='Directory with generated configurations')
    results_parser.add_argument('--model', required=True, help='Model name')
    results_parser.add_argument('--path', '-p', required=True, help='Path to logs directory')
    results_parser.add_argument('--log-prefix', required=True, help='Log file prefix')
    results_parser.add_argument('--top-n', type=int, default=10, help='Number of top configurations to display')
    results_parser.add_argument('--force-reconstruct', action='store_true', help='Force reconstruction')
    results_parser.add_argument("--cost-per-gpu-hour", type=float, default=24.0, help="Cost per GPU hour")
    results_parser.add_argument('--quiet', action='store_true', help='Suppress output')
    results_parser.add_argument('--launcher-node-group', required=True, help='Node group for launcher jobs (CPU jobs)')
    results_parser.add_argument('--training-node-group', required=True, help='Node group for training jobs (GPU jobs)')
    
    # List-configs command
    list_configs_parser = subparsers.add_parser('list-configs', help='List generated configurations')
    list_configs_parser.add_argument('--config-dir', required=True, help='Directory with generated configurations')
    list_configs_parser.add_argument('--model', required=True, help='Model name')
    list_configs_parser.add_argument('--launcher-node-group', required=True, help='Node group for launcher jobs (CPU jobs)')
    list_configs_parser.add_argument('--training-node-group', required=True, help='Node group for training jobs (GPU jobs)')

    subparsers.add_parser('list-models', help='List supported models')
    
    return parser

def parse_list_arg(value):
    """Parse comma-separated list argument."""
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    return value

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle global options first
    if args.list_node_groups:
        print("Listing available node groups...")
        list_available_node_groups()
        return
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'generate':
            kwargs = {
                'model': args.model,
                'nodes': args.nodes,
                'gpus_per_node': args.gpus_per_node,
                'config_dir': args.config_dir,
                'mount_path': args.mount_path,
                'mount_from': args.mount_from,
                'launcher_node_group': args.launcher_node_group,
                'training_node_group': args.training_node_group,
                'logs_subdir': args.logs_subdir,
                'resource_shape': args.resource_shape,
                'tensor_parallel_sizes': parse_list_arg(args.tensor_parallel_sizes),
                'pipeline_parallel_sizes': parse_list_arg(args.pipeline_parallel_sizes),
                'context_parallel_sizes': parse_list_arg(args.context_parallel_sizes),
                'expert_parallel_sizes': parse_list_arg(args.expert_parallel_sizes),
                'virtual_pipeline_parallel_sizes': parse_list_arg(args.virtual_pipeline_parallel_sizes),
                'global_batch_sizes': parse_list_arg(args.global_batch_sizes),
                'micro_batch_sizes': parse_list_arg(args.micro_batch_sizes),
                'max_steps_per_run': args.max_steps_per_run,
                'seq_length': args.seq_length,
                'num_tokens_in_b': args.num_tokens_in_b,
                'container_image': args.container_image,
            }
            launch_generate_remote(kwargs)
            
        elif args.command == 'run':
            launch_run_remote(args.config_dir, args.model, args.sequential, args.run_all, args.launcher_node_group, args.training_node_group, args.mount_from)
            
        elif args.command == 'results':
            launch_results_remote(
                args.config_dir, args.model, args.path, args.log_prefix,
                args.top_n, args.force_reconstruct, args.cost_per_gpu_hour, args.quiet, args.launcher_node_group, args.training_node_group, args.mount_from
            )
            
        elif args.command == 'list-configs':
            launch_list_configs_remote(args.config_dir, args.model, args.launcher_node_group, args.training_node_group, args.mount_from)
            
        elif args.command == 'list-models':
            list_models()
            
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error executing {args.command}: {e}")
        print("\nDebugging tips:")
        print("1. Check available node groups: python launcher.py --list-node-groups")
        print("2. Try without node group specification")
        print("3. Check Lepton workspace permissions")
        sys.exit(1)

if __name__ == "__main__":
    main() 