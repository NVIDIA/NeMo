import os
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table

from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from nemo.collections.llm.tools.autotuner.core.utils import extract_all_values

console = Console()


def _display_memory_analysis(memory_analysis):
    if not memory_analysis:
        console.print("[yellow]No memory analysis available[/yellow]")
        return
    console.print(f"\n[cyan] CUDA Memory Analysis & Run Status[/cyan]")
    table = Table(show_header=True, show_lines=True, title="Memory Usage Analysis & Execution Status")
    table.add_column("Configuration", style="cyan", width=40)
    table.add_column("Memory Status", style="white", width=12)
    table.add_column("Run Status", style="white", width=12)
    table.add_column("Est. Usage (GB)", style="blue", width=15)
    table.add_column("GPU Memory (GB)", style="green", width=15)
    oom_count = 0
    safe_count = 0
    for config_name, analysis in memory_analysis.items():
        will_oom = analysis.get('will_oom', False)
        usage_gb = analysis.get('estimated_usage_gb', 0)
        total_gb = analysis.get('total_gpu_memory_gb', 0)
        config_values = analysis.get('config_values', {})
        if will_oom:
            memory_status = "[red]⚠ OOM Risk[/red]"
            run_status = "[red]Skip[/red]"
            oom_count += 1
        else:
            memory_status = "[green]Safe[/green]"
            run_status = "[green]▶ Run[/green]"
            safe_count += 1
        if config_name != "base_config" and all(
            v in [1, 512, 8192]
            for v in [config_values.get('tp', 1), config_values.get('mbs', 1), config_values.get('gbs', 512)]
        ):
            config_values = extract_all_values(config_name)
        table.add_row(config_name, memory_status, run_status, f"{usage_gb:.1f}", f"{total_gb:.0f}")
    console.print(table)
    console.print(f"\n[cyan]Memory Analysis Summary:[/cyan]")
    console.print(f"Safe configurations (will run): {safe_count}")
    console.print(f"Potential OOM configurations (will be skipped): {oom_count}")
    if oom_count > 0:
        console.print(f"\n[yellow]⚠ Warning: {oom_count} configurations will be SKIPPED during autotune run[/yellow]")
        console.print("[yellow]These configurations may cause CUDA OOM errors[/yellow]")
        console.print("[blue]To run them anyway: use run_all=True[/blue]")
        console.print("[blue] To fix: reduce micro batch sizes or increase parallelism[/blue]")


def _display_configs_table(config_dir, model_name=None):
    try:
        if not model_name:
            model_name = os.path.basename(config_dir)
        args_file_path = os.path.join(config_dir, "args.json")
        if os.path.exists(args_file_path):
            args = AutoTuneArgs.load_from_file(args_file_path)
            metadata = args.metadata
            has_metadata = bool(metadata)
        else:
            console.print(f"[yellow]No args.json found in {config_dir}[/yellow]")
            metadata = {}
            has_metadata = False
            args = None
    except Exception as e:
        console.print(f"[yellow]Could not load metadata: {e}[/yellow]")
        metadata = {}
        has_metadata = False
        args = None
    all_files = os.listdir(config_dir)
    json_files = [f for f in all_files if f.endswith('.json') and f not in ['args.json']]
    if not json_files:
        console.print(f"[yellow]No configuration files found in: {config_dir}[/yellow]")
        return
    base_config_matches = metadata.get('base_config_matches', [])
    config_names = metadata.get('config_names', [])
    num_configs_generated = metadata.get('num_configs_generated', len(json_files) - 1)
    total_gpus = metadata.get('total_gpus', 'Unknown')
    generation_timestamp = metadata.get('generation_timestamp', 'Unknown')
    table = Table(show_header=True, show_lines=True, title=f"Configuration Files - {model_name or 'Unknown Model'}")
    table.add_column("Filename", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Size", style="white")
    for filename in sorted(json_files):
        filepath = os.path.join(config_dir, filename)
        try:
            stat = os.stat(filepath)
            size = f"{stat.st_size:,} bytes"
        except Exception as e:
            size = f"[red]Error: {e}[/red]"
        if filename == "base_config.json":
            if base_config_matches:
                status = f"[yellow]Base Config (equivalent to: {', '.join(base_config_matches)})[/yellow]"
            else:
                status = "[bold green]Base Config[/bold green]"
        else:
            config_name = filename.replace('.json', '')
            if has_metadata:
                if config_name in base_config_matches:
                    status = "[blue]Base Config Match[/blue]"
                elif config_name in config_names:
                    status = "[green]Generated[/green]"
                else:
                    status = "[dim]Unknown[/dim]"
            else:
                status = "[green]Generated[/green]"
        table.add_row(filename, status, size)
    console.print(table)
    console.print(f"\n[cyan]Summary:[/cyan]")
    if has_metadata:
        console.print(f"Model: {model_name}")
        console.print(f"Total GPUs: {total_gpus}")
        if args and hasattr(args, 'global_batch_sizes'):
            console.print(f"Global batch sizes: {args.global_batch_sizes}")
        if args and hasattr(args, 'resource_shape'):
            console.print(f"Resource shape: {args.resource_shape}")
        console.print(f"Generated configurations: {num_configs_generated}")
        console.print(f"Base config matches: {len(base_config_matches)}")
        console.print(f"Configuration files: {len(json_files)}")
        if generation_timestamp != 'Unknown':
            console.print(f"Generated: {generation_timestamp}")
        if args and hasattr(args, 'has_memory_analysis') and args.has_memory_analysis():
            memory_analysis = args.get_memory_analysis()
            _display_memory_analysis(memory_analysis)
            oom_configs = [name for name, analysis in memory_analysis.items() if analysis.get("will_oom", False)]
            if oom_configs:
                console.print(f"\n[yellow]Run Behavior Notes:[/yellow]")
                console.print(
                    f"  • By default, autotune run will SKIP the {len(oom_configs)} flagged configuration(s)"
                )
                console.print(f"  • Use run_all=True to run ALL configurations including potential OOM ones")
        if args and hasattr(args, 'has_performance_results') and args.has_performance_results():
            results_timestamp = metadata.get('results_timestamp', 'Unknown')
            performance_dict = args.get_performance_dict()
            console.print(f"Performance Results: Available ({len(performance_dict)} configs)")
            if results_timestamp != 'Unknown':
                console.print(f"Results analyzed: {results_timestamp}")
            console.print("[green] Use analyse_results() to analyze performance[/green]")
        else:
            console.print("[yellow]Performance Results: Not available[/yellow]")
            console.print("[yellow]Run results() to generate performance data[/yellow]")
        if base_config_matches:
            console.print(f"\n[yellow]Note:[/yellow] Base config is equivalent to: {', '.join(base_config_matches)}")
            console.print("[yellow]These configurations will not be run separately during training.[/yellow]")
    else:
        console.print(f"Configuration files: {len(json_files)}")
        console.print("[yellow]No metadata available. Re-run generate() for detailed status.[/yellow]")


def display_performance_analysis(analysis_data: Optional[Dict[str, Any]]) -> None:
    if not analysis_data or not analysis_data.get('sorted_configs'):
        console.print("[yellow]No performance data to analyze[/yellow]")
        return
    config_analysis = analysis_data['config_analysis']
    sorted_configs = analysis_data['sorted_configs']
    args = analysis_data['args']
    best_config_name, best_config = sorted_configs[0]
    worst_config_name, worst_config = sorted_configs[-1]
    base_config_generated_name = args.metadata.get('base_config_generated_name', None)
    base_config_matches = args.metadata.get('base_config_matches', [])
    base_config_name = None
    base_config = None
    if base_config_generated_name and base_config_generated_name in config_analysis:
        base_config_name = base_config_generated_name
        base_config = config_analysis[base_config_generated_name]
    elif 'base_config' in config_analysis:
        base_config_name = 'base_config'
        base_config = config_analysis['base_config']

    console.print("\n[cyan] Performance & Cost Analysis Summary[/cyan]")
    console.print("=" * 80)
    console.print(f"\n[green]Best Performing Configuration: {best_config_name}[/green]")
    console.print(f"  M-TFLOPs/GPU: {best_config.get('m_tflops_gpu', 'N/A'):.2f}")
    console.print(f"  Time per Global Step: {best_config.get('time_per_global_step', 'N/A'):.4f}s")
    console.print(f"  Total Training Time: {best_config.get('total_training_time_days', 'N/A'):.1f} days")
    console.print(f"  Total Training Cost: ${best_config.get('total_cost', 'N/A'):,.2f}")
    if base_config and base_config_name != best_config_name:
        console.print(f"\n[blue]Base Configuration: {base_config_name}[/blue]")
        console.print(f"  M-TFLOPs/GPU: {base_config.get('m_tflops_gpu', 'N/A'):.2f}")
        console.print(f"  Time per Global Step: {base_config.get('time_per_global_step', 'N/A'):.4f}s")
        console.print(f"  Total Training Time: {base_config.get('total_training_time_days', 'N/A'):.1f} days")
        console.print(f"  Total Training Cost: ${base_config.get('total_cost', 'N/A'):,.2f}")
        tflops_improvement = (
            (best_config.get('m_tflops_gpu', 0) - base_config.get('m_tflops_gpu', 0))
            / base_config.get('m_tflops_gpu', 1)
        ) * 100
        time_savings = base_config.get('total_training_time_hours', 0) - best_config.get(
            'total_training_time_hours', 0
        )
        cost_savings = base_config.get('total_cost', 0) - best_config.get('total_cost', 0)
        cost_savings_percent = (cost_savings / base_config.get('total_cost', 1)) * 100
        console.print(f"\n[yellow]Best vs Base Performance & Cost Savings:[/yellow]")
        console.print(f"  M-TFLOPs/GPU improvement: {tflops_improvement:+.1f}%")
        console.print(f"  Training time savings: {time_savings:.1f} hours ({time_savings/24:.1f} days)")
        console.print(f"  Cost savings: ${cost_savings:,.2f} ({cost_savings_percent:+.1f}%)")
        if cost_savings > 0:
            console.print(f"  [green] Total Savings: ${cost_savings:,.2f}[/green]")
        else:
            console.print(f"  [red] Additional Cost: ${abs(cost_savings):,.2f}[/red]")
    if worst_config_name != best_config_name:
        console.print(f"\n[red]Worst Performing Configuration: {worst_config_name}[/red]")
        console.print(f"  M-TFLOPs/GPU: {worst_config.get('m_tflops_gpu', 'N/A'):.2f}")
        console.print(f"  Time per Global Step: {worst_config.get('time_per_global_step', 'N/A'):.4f}s")
        console.print(f"  Total Training Time: {worst_config.get('total_training_time_days', 'N/A'):.1f} days")
        console.print(f"  Total Training Cost: ${worst_config.get('total_cost', 'N/A'):,.2f}")
        time_diff = worst_config.get('total_training_time_hours', 0) - best_config.get('total_training_time_hours', 0)
        cost_diff = worst_config.get('total_cost', 0) - best_config.get('total_cost', 0)
        tflops_diff = (
            (best_config.get('m_tflops_gpu', 0) - worst_config.get('m_tflops_gpu', 0))
            / worst_config.get('m_tflops_gpu', 1)
        ) * 100
        console.print(f"\n[yellow] Best vs Worst Performance & Cost Difference:[/yellow]")
        console.print(f"  M-TFLOPs/GPU difference: {tflops_diff:+.1f}%")
        console.print(f"  Training time difference: {time_diff:.1f} hours ({time_diff/24:.1f} days)")
        console.print(f"  Cost difference: ${cost_diff:,.2f}")
        console.print(f"  [red]Potential waste with worst config: ${cost_diff:,.2f}[/red]")
    console.print(f"\n[cyan] Top 5 Configurations - Performance & Cost Analysis[/cyan]")
    table = Table(show_header=True, show_lines=True, title="Performance & Cost Ranking")
    table.add_column("Rank", style="yellow", width=6)
    table.add_column("Configuration", style="cyan", width=120)
    table.add_column("M-TFLOPs/GPU", style="green", width=12)
    table.add_column("Training Days", style="blue", width=12)
    table.add_column("Total Cost", style="red", width=12)
    table.add_column("Status", style="white", width=15)
    for i, (config_name, config_data) in enumerate(sorted_configs[:5], 1):
        status = "Generated"
        if config_name in base_config_matches or config_name == 'base_config':
            status = "Base Config"
        elif i == 1:
            status = " Best"
        table.add_row(
            str(i),
            config_name,
            f"{config_data.get('m_tflops_gpu', 0):.2f}",
            f"{config_data.get('total_training_time_days', 0):.1f}",
            f"${config_data.get('total_cost', 0):,.0f}",
            status,
        )
    console.print(table)
    console.print(f"\n[cyan] Cost Efficiency Analysis[/cyan]")
    console.print("=" * 50)
    most_efficient = min(config_analysis.items(), key=lambda x: x[1].get('cost_per_tflop', float('inf')))
    most_efficient_name, most_efficient_data = most_efficient
    console.print(f"Most Cost-Efficient: {most_efficient_name}")
    console.print(f"  Total Cost: ${most_efficient_data.get('total_cost', 'N/A'):,.2f}")
    console.print(f"  M-TFLOPs/GPU: {most_efficient_data.get('m_tflops_gpu', 'N/A'):.2f}")
    console.print(f"\n[cyan] Recommendations[/cyan]")
    console.print("=" * 40)
    console.print(f"Best Performance: '{best_config_name}'")
    console.print(f"Most Cost-Efficient: '{most_efficient_name}'")
    if base_config:
        if base_config_name != best_config_name:
            savings = base_config.get('total_cost', 0) - best_config.get('total_cost', 0)
            console.print(f"Switch from base config to save: ${savings:,.2f}")
        else:
            console.print(f"Base config is already optimal!")
    console.print("\n[green]Cost analysis completed successfully![/green]")
