# AutoTuner

AutoTuner is a fully automated, predictive configuration builder and orchestrated tool that achieves maximum model throughput for LLM Training.

## Overview

AutoTuner extends NeMo's Auto Configurator with a fully automated pipeline that searches for hyperparameters achieving maximum training throughput while automatically preventing CUDA OOM errors and providing comprehensive performance & cost analysis for Large Language Models. Unlike research-only tools, AutoTuner runs directly on your GPU infrastructure, making it ready for production deployment and real-world training optimization.

### Note

AutoTuner is supported for all NeMo models including GPT-based models: GPT3, LLama, Mixtral, Mistral, Gemma, Nemotron, Starcoder, Qwen, and more.

### AutoTuner Capabilities

AutoTuner is designed to iterate over different model configurations quickly and find the best configuration that maximizes training throughput while minimizing both time and money. It offers comprehensive features to facilitate this, as detailed below.

- **CUDA OOM Prevention**: Automatically detects and filters out configurations that would cause out-of-memory errors, preventing training failures
- **Orchestrated Pipeline**: Fully automated workflow from configuration generation and pre-training to results analysis.
- **Rigorous Performance Analysis**: Comprehensive metrics including TFLOPS, training time, and cost optimization.
- **Rich Visualization**: Descriptive tables and charts for easy decision making and result interpretation.
- **Production Deployment**: Runs directly on customer GPU infrastructure, ready for enterprise use.



# NeMo Autotuner Launcher

A tool for orchestrating NeMo autotuner workflows on remote clusters using Lepton. This launcher handles packaging local code and data, then executes autotuner steps remotely.

## Quick Start

### Prerequisites
- Installation

   ```bash
   # Ensure you have NeMo, nemo_run and rich installed
   pip install nemo nemo_run rich
   ```
- NeMo workspace mounted at `/nemo-workspace`
- Access to GPU resources on Lepton

## Basic Usage

### 1. Generate Configurations (`generate`)

Creates optimized training configurations for your model and infrastructure.
**Example**

```bash
python launcher.py generate \
  --model llama2_7b \                                                       # Model name to generate configs for
  --nodes 4 \                                                               # Number of compute nodes
  --gpus-per-node 8 \                                                       # GPUs per node
  --config-dir nemo-workspace/autotuner-data/autotuner-configs \            # Directory to save generated configs
  --mount-path /workspace \                                                 # Remote mount path for workspace
  --mount-from node-nfs:shared \                                            # Mount source for shared storage
  --node-group gpu-cluster \                                                # Node group for resource allocation
  --logs-subdir logs \                                                      # Subdirectory for training logs
  --resource-shape gpu.8xh200 \                                             # GPU resource shape for training
  --tensor-parallel-sizes 1,2,4,8 \                                         # Tensor parallel sizes (comma-separated)
  --pipeline-parallel-sizes 1,2,4 \                                         # Pipeline parallel sizes (comma-separated)
  --context-parallel-sizes 1,2 \                                            # Context parallel sizes (comma-separated)
  --expert-parallel-sizes 1 \                                               # Expert parallel sizes (comma-separated)
  --virtual-pipeline-parallel-sizes 1,2 \                                   # Virtual pipeline parallel sizes (comma-separated)
  --global-batch-sizes 256,512,1024 \                                       # Global batch sizes (comma-separated)
  --micro-batch-sizes 1,2,4,8 \                                             # Micro batch sizes (comma-separated)
  --max-steps-per-run 10 \                                                  # Maximum steps per training run
  --seq-length 8192 \                                                       # Sequence length for training
  --num-tokens-in-b 15000 \                                                 # Number of tokens in billions
  --container-image nvcr.io/nvidia/nemo:25.04                               # Container image to use
```

**Expected output:**
```
You can train a 2B parameter model in 2480.16 days using 8 GPUs. This result assumes you are training to 15000B tokens, and each GPU achieves 140 TFLOPS.
Valid config: SeqLen=8192, GBS=256, MBS=1, TP=1, PP=1, CP=1, EP=1, VP=None. Adding to directory.
Valid config: SeqLen=8192, GBS=512, MBS=1, TP=1, PP=1, CP=1, EP=1, VP=None. Adding to directory.
Valid config: SeqLen=8192, GBS=256, MBS=2, TP=1, PP=1, CP=1, EP=1, VP=None. Adding to directory.
Valid config: SeqLen=8192, GBS=512, MBS=2, TP=1, PP=1, CP=1, EP=1, VP=None. Adding to directory.

All candidate configurations created correctly. Total number of configs: 4.

Generated configurations successfully
Configurations for model: gemma2_2b
Location: /autotuner/generated_configs/gemma2_2b
                                    Configuration Files - gemma2_2b                                  
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Filename                                                                ┃ Status      ┃ Size        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ base_config.json                                                        │ Base Config │ 7,669 bytes │
├─────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ gemma_2b_1nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_256.json │ Unknown     │ 7,454 bytes │
├─────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ gemma_2b_1nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_512.json │ Unknown     │ 7,454 bytes │
├─────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ gemma_2b_1nodes_tp_1_pp_1_cp_1_ep_1_mbs_2_vp_None_seq_8192_gbs_256.json │ Unknown     │ 7,454 bytes │
└─────────────────────────────────────────────────────────────────────────┴─────────────┴─────────────┘

 CUDA Memory Analysis & Run Status
                                   Memory Usage Analysis & Execution Status                                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃                                          ┃ Memory       ┃              ┃                 ┃                 ┃
┃ Configuration                            ┃ Status       ┃ Run Status   ┃ Est. Usage (GB) ┃ GPU Memory (GB) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ base_config                              │ Safe         │ ▶ Run        │ 19.2            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤
│ gemma_2b_1nodes_tp_1_pp_1_cp_1_ep_1_mbs… │ Safe         │ ▶ Run        │ 21.9            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤
│ gemma_2b_1nodes_tp_1_pp_1_cp_1_ep_1_mbs… │ Safe         │ ▶ Run        │ 21.9            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤
│ gemma_2b_1nodes_tp_1_pp_1_cp_1_ep_1_mbs… │ Safe         │ ▶ Run        │ 27.4            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤

Memory Analysis Summary:
Safe configurations (will run): 67
Potential OOM configurations (will be skipped): 0
Performance Results: Not available
Run results() to generate performance data

```
### 2. Run Experiments (`run`)

Executes training experiments using generated configurations.

**Example:**
```bash
python launcher.py run \
  --config-dir nemo-workspace/autotuner-data/autotuner-configs \           # Directory containing generated configs
  --model llama2_7b \                                                      # Model name to run experiments for
  --sequential \                                                            # Run experiments one at a time (vs parallel)
  --run-all                                                                # Run all configs (vs CUDA safe ones only)
```

### 3. Gather and Analyze Results (`results`)

Analyzes training results and generates performance reports.

**Example:**
```bash
python launcher.py results \
  --config-dir nemo-workspace/autotuner-data/autotuner-configs \           # Directory containing model configs
  --model llama2_7b \                                                      # Model name for analysis
  --path ./training_logs \                                                 # Path to training log files
  --log-prefix nemo \                                                      # Prefix for log files to analyze
  --top-n 5 \                                                              # Number of top configs to analyze
  --force-reconstruct \                                                    # Force rebuild analysis from logs
  --cost-per-node-hour 24.0 \                                              # Cost per node-hour for cost analysis
  --quiet                                                                  # Suppress verbose output
```
**Expected output:**
```
Top 5 configs sorted from fastest to slowest:
Config #1 - llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_256: 604.27 TFLOPS per GPU with 11.3500s per global step.
Config #2 - llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_2_vp_None_seq_8192_gbs_256: 602.68 TFLOPS per GPU with 11.3800s per global step.
Config #3 - llama_7b_2nodes_tp_1_pp_2_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_256: 572.49 TFLOPS per GPU with 11.9800s per global step.
Config #4 - llama_7b_2nodes_tp_1_pp_2_cp_1_ep_1_mbs_2_vp_None_seq_8192_gbs_256: 558.51 TFLOPS per GPU with 12.2800s per global step.
Config #5 - llama_7b_2nodes_tp_2_pp_1_cp_1_ep_1_mbs_2_vp_None_seq_8192_gbs_256: 545.19 TFLOPS per GPU with 12.5800s per global step.

==================================================
Optimal config: llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_256 with 11.3500s per global step.
==================================================

 Performance & Cost Analysis Summary
================================================================================

Best Performing Configuration: llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_512
  M-TFLOPs/GPU: 612.91
  Time per Global Step: 22.3800s
  Total Training Time: 926.4 days
  Total Training Cost: $133,395.20

Best vs Base Performance & Cost Savings:
  M-TFLOPs/GPU improvement: +14.8%
  Training time savings: 3288.2 hours (137.0 days)
  Cost savings: $19,729.14 (+12.9%)

 Top 5 Configurations - Performance & Cost Analysis
                                              Performance & Cost Ranking                                              
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃   ┃                                                                   ┃          ┃ Traini… ┃ Total    ┃            ┃
┃ … ┃ Configuration                                                     ┃ M-TFLOP… ┃ Days    ┃ Cost     ┃ Status     ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1 │ llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_5… │ 612.91   │ 926.4   │ $133,395 │  Best      │
├───┼───────────────────────────────────────────────────────────────────┼──────────┼─────────┼──────────┼────────────┤
│ 2 │ llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_2… │ 604.27   │ 939.6   │ $135,303 │ Generated  │
├───┼───────────────────────────────────────────────────────────────────┼──────────┼─────────┼──────────┼────────────┤
│ 3 │ llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_2_vp_None_seq_8192_gbs_2… │ 602.68   │ 942.1   │ $135,660 │ Generated  │
├───┼───────────────────────────────────────────────────────────────────┼──────────┼─────────┼──────────┼────────────┤
│ 4 │ llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_2_vp_None_seq_8192_gbs_5… │ 602.41   │ 942.5   │ $135,720 │ Generated  │
├───┼───────────────────────────────────────────────────────────────────┼──────────┼─────────┼──────────┼────────────┤
│ 5 │ llama_7b_2nodes_tp_1_pp_2_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_5… │ 580.00   │ 978.9   │ $140,965 │ Generated  │
└───┴───────────────────────────────────────────────────────────────────┴──────────┴─────────┴──────────┴────────────┘

 Recommendations
========================================
Best Performance: 'llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_512'
Most Cost-Efficient: 'llama_7b_2nodes_tp_1_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_512'
Switch from base config to save: $19,729.14

Cost analysis completed successfully!
```

### 4. List Configurations (`list-configs`)

Lists available configurations for a model.

**Example:**
```bash
python launcher.py list-configs \
  --config-dir nemo-workspace/autotuner-data/autotuner-configs \     # Directory containing generated configs
  --model llama2_7b                                                    # Model name to list configs for
```

### 5. List Models (`list-models`)

Lists all supported models.

```bash
python launcher.py list-models
```

### Remote Execution

All operations are executed remotely on Lepton clusters:

1. **Code Packaging**: Local scripts are packaged and uploaded
2. **Job Creation**: Lepton jobs are created with appropriate resources
3. **Log Streaming**: Real-time logs are streamed to your terminal
4. **Result Retrieval**: Results are available in the mounted workspace

### One-time Setup

The launcher automatically handles one-time setup tasks:

- **NeMo Installation**: Local NeMo changes are installed once per workspace
- **Environment Setup**: Python paths and cache directories are configured
- **Dependency Management**: All required packages are available in the base image


## 🐛 Troubleshooting

### Common Issues

1. **Job Fails Immediately**
   - Check Lepton authentication: `lepton auth status`
   - Verify resource availability: `lepton resource list`
   - Check workspace mounting: Ensure `/nemo-workspace` is accessible

2. **Configuration Generation Fails**
   - Verify model name is supported: `python launcher.py list-models`
   - Check resource requirements match available resources
   - Ensure config directory is writable

3. **Training Experiments Fail**
   - Check GPU availability and compatibility
   - Verify data paths and permissions
   - Review logs for specific error messages

4. **Results Analysis Issues**
   - Ensure log files exist and are readable
   - Check log prefix matches actual log files
   - Verify cost parameters are reasonable

### Getting Help

```bash
# Show general help
python launcher.py --help

# Show help for specific command
python launcher.py generate --help
python launcher.py run --help
python launcher.py results --help
```

### Log Retrieval

If a job fails, you can retrieve logs manually:

```bash
# Get job logs
lepton job logs JOB_ID

# List recent jobs
lepton job list

# Get job details
lepton job get JOB_ID

### Step 2: Run Training Experiments
```bash
# Edit run_experiments.py with correct paths
python3 run_experiments.py
```


## Configuration Tips for Scripts

### Model Selection
- Use exact NeMo model names: `"llama2"`, `"mixtral_8x7b"`, `"gemma2_2b"`, etc.
- Check supported models with: 
    ```
    from nemo.collections.llm.tools.autotuner import list_models; list_models()
    ```

### Resource Configuration
- `resource_shape`: Use format `"gpu.countxtype"` (e.g., `"gpu.8xh200"`, `"gpu.4xa100"`)
- Ensure `nodes * gpus_per_node` matches your total GPU count
- Parallelism sizes must divide evenly into total GPU count

### Memory Considerations
- AutoTuner automatically detects potential OOM configurations
- Configurations flagged as OOM risks will be skipped during training
- Use `run_all=True` to force run all configurations (not recommended)

### Performance Optimization
- Start with smaller parameter ranges for quick testing
- Increase `max_steps_per_run` for more accurate performance measurement
- Use `top_n=10` in results analysis to see more configurations

### Path Configuration
- Update all file paths in scripts to match your directory structure
- Ensure `config_dir` and `logs_subdir` are absolute paths or relative to script location
- The `args.json` file will be saved in `config_dir/model_name/args.json`

## Configuration Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | NeMo model name (e.g., "llama2", "mixtral_8x7b") |
| `nodes` | int | Number of nodes |
| `gpus_per_node` | int | GPUs per node |
| `resource_shape` | str | GPU type and count (e.g., "gpu.8xh200") |
| `config_dir` | str | Directory to save configurations |
| `logs_subdir` | str | Directory to save training logs |

### Parallelism Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_batch_sizes` | List[int] | [512] | Global batch sizes to test |
| `tensor_parallel_sizes` | List[int] | [1, 2] | Tensor parallelism sizes |
| `pipeline_parallel_sizes` | List[int] | [1, 2] | Pipeline parallelism sizes |
| `context_parallel_sizes` | List[int] | [1, 2] | Context parallelism sizes |
| `expert_parallel_sizes` | List[int] | [1] | Expert parallelism sizes |
| `micro_batch_sizes` | List[int] | [1, 2] | Micro batch sizes |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps_per_run` | int | 10 | Steps per experiment |
| `max_steps` | int | 10 | Total training steps |
| `seq_length` | int | 8192 | Sequence length |
| `num_tokens_in_b` | int | 15000 | Tokens in billions |

### Infrastructure Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mount_path` | str | "/nemo-workspace" | Container mount path |
| `mount_from` | str | "node-nfs:shared" | Storage backend |
| `node_group` | str | None | Node group name |
| `container_image` | str | "nvcr.io/nvidia/nemo:25.04" | Container image |

## Output Files

### Generated Files

- `config_dir/model_name/base_config.json` - Base configuration
- `config_dir/model_name/config_*.json` - Generated configurations
- `config_dir/model_name/args.json` - Arguments and metadata
- `logs_subdir/model_name/` - Training logs for each configuration

### Results Analysis

The `results()` function displays:
- **Performance Summary**: Best/worst/base configurations
- **Cost Analysis**: Training time and cost comparisons
- **Top Configurations**: Ranked by performance
- **Recommendations**: Best performance vs cost efficiency

## Troubleshooting

### Common Issues with Scripts

1. **Import Errors**: Ensure NeMo and nemo_run are properly installed
   ```bash
   pip install nemo nemo_run rich
   ```

2. **Path Issues**: Update all file paths in scripts to match your directory structure
   - Ensure paths are always absolute

3. **Configuration Validation Failed**: Review parallelism constraints
   - Ensure `nodes * gpus_per_node` matches total GPU count
   - Parallelism sizes must divide evenly into total GPU count

4. **No Configurations to Run**: 
   - All configs flagged for OOM
   - Reduce `micro_batch_sizes` or increase parallelism
   - Use `run_all=True` to force run all configurations (not recommended)

5. **Logs Directory Not Found**: Run training experiments first
   - Ensure `run_experiments.py` completes successfully before running `results.py`

### Memory Issues

- Reduce `micro_batch_sizes` or increase parallelism
- Use `run_all=True` to run OOM-risk configurations anyway
- Check GPU memory specifications in `resource_shape`

### Performance Issues

- Increase `max_steps_per_run` for more accurate timing
- Use `force_reconstruct=True` to regenerate configurations
- Check log files for training errors


This README provides a comprehensive guide to using the AutoTune module step-by-step, with examples and troubleshooting tips.
