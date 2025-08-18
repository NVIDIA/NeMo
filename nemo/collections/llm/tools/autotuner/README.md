# AutoTuner

AutoTuner is a fully automated, cost-aware, customized and intelligent model configuration and throughput optimization for LLM pre-training workloads

### Note

AutoTuner supports all NeMo adapted models. 

### AutoTuner Capabilities

AutoTuner is designed to iterate over different model configurations quickly and find the best configuration that maximizes training throughput while minimizing both time and money. It offers comprehensive features to facilitate this, as detailed below.

- **CUDA OOM Prevention**: Automatically detects and filters out configurations that would cause out-of-memory errors, preventing training failures
- **Orchestrated Pipeline**: Fully automated workflow from configuration generation and pre-training to results analysis.
- **Rigorous Performance Analysis**: Comprehensive metrics including TFLOPS, training time, and cost optimization.
- **Rich Visualization**: Descriptive tables and charts for easy decision making and result interpretation.
- **Production Deployment**: Runs directly on customer GPU infrastructure, ready for enterprise use.

# NeMo AutoTuner Launcher

A tool for orchestrating NeMo AutoTuner workflows on remote clusters using DGX Cloud Lepton. This launcher handles packaging local code and data, then executes AutoTuner steps remotely.

## Quick Start

### Prerequisites
- Installation

   ```bash
   # Ensure you have NeMo, nemo_run and rich installed
   pip install nemo_toolkit[nlp]
   pip install git+https://github.com/NVIDIA-NeMo/Run.git
   pip install rich
   ```
- NeMo workspace mounted at `/nemo-workspace`
- Access to GPU resources on DGX Cloud Lepton.

### Environment Setup

**Required Environment Variables:**

Before running any AutoTuner commands, you must authenticate to lepton and also set these environment variables:

```bash
# Login to your lep workspace
lep login -c <workspace-id>:<workspace-token>
```

```bash
# Set your Lepton workspace credentials
export LEPTON_AUTOTUNER_WORKSPACE_ID="your_workspace_id_here"
export LEPTON_AUTOTUNER_TOKEN="your_workspace_token_here"
```

**How to get these values:**

1. **Get your workspace ID and token:**
   ```bash
   lep workspace list
   ```

3. **Verify the setup:**
   ```bash
   echo "Workspace ID: $LEPTON_AUTOTUNER_WORKSPACE_ID"
   echo "Token: $LEPTON_AUTOTUNER_TOKEN"
   ```

**Important Notes:**
- These environment variables are **required** for all AutoTuner commands
- The launcher uses these credentials to authenticate with Lepton and launch remote jobs
- Without these variables set, you'll get authentication errors
- You can add these to your shell profile (`.bashrc`, `.zshrc`) for persistence

## Basic Usage

### 1. Generate Configurations (`generate`)

Creates optimized training configurations for your model and infrastructure.

```bash
python launcher.py generate \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model gemma2_9b \
  --launcher-node-group tme-nebius-h200-01 \
  --training-node-group tme-nebius-h200-01 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace \
  --resource-shape gpu.8xh200  \
  --nodes 8 \
  --gpus-per-node 8 \
  --seq-length 8192 \
  --num-tokens-in-b 1000 \
  --global-batch-sizes 256 \
  --tensor-parallel-sizes 2,1 \
  --pipeline-parallel-sizes 2,1 \
  --virtual-pipeline-model-parallel-sizes 1,2 \
  --max-model-parallel-size 64 \
  --context-parallel-sizes 1 \
  --expert-parallel-sizes 1 \
  --micro-batch-sizes 1,2 \
  --max-steps-per-run 10 \
  --max-steps 10 \
  --logs-subdir /nemo-workspace/autotuner/new/logs \
  --container-image nvcr.io/nvidia/nemo:25.07
```



**Expected output:**
```
You can train a 9B parameter model in 2480.16 days using 64 GPUs. This result assumes you are training to 1000B tokens, and each GPU achieves 140 TFLOPS.
Valid config: SeqLen=8192, GBS=256, MBS=1, TP=2, PP=2, CP=1, EP=1, VP=None. Adding to directory.
Valid config: SeqLen=8192, GBS=512, MBS=1, TP=2, PP=2, CP=1, EP=1, VP=None. Adding to directory.
Valid config: SeqLen=8192, GBS=256, MBS=2, TP=2, PP=2, CP=1, EP=1, VP=None. Adding to directory.
Valid config: SeqLen=8192, GBS=512, MBS=2, TP=2, PP=2, CP=1, EP=1, VP=None. Adding to directory.

Metadata and objects saved to: 
/nemo-workspace/autotuner/new/generated_configs/gemma2_9b/args.json
Configurations generated successfully with performance optimizations!
Saved to: /nemo-workspace/autotuner/new/generated_configs/gemma2_9b
Generated 12 configurations

Memory Analysis Summary:
Configurations that will run safely: 13
Use 'lep autotune list-configs' to see detailed memory analysis
```

### 2. List Configurations (`list-configs`)
```bash
python launcher.py list-configs \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model gemma2_9b \
  --launcher-node-group tme-nebius-h200-01 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace
```
**Expected output:**
```
Configurations for model: gemma2_9b
Location: /nemo-workspace/autotuner/new/generated_configs/gemma2_9b

                                    Configuration Files - gemma2_9b                                  
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Filename                                                                ┃ Status      ┃ Size        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ base_config.json                                                        │ Base Config │ 7,669 bytes │
├─────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ gemma2_9b_8nodes_tp_2_pp_2_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_256.json │ Generated     │ 7,454 bytes │
├─────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ gemma2_9b_8nodes_tp_2_pp_2_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_512.json │ Generated     │ 7,454 bytes │
├─────────────────────────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ gemma2_9b_8nodes_tp_2_pp_2_cp_1_ep_1_mbs_2_vp_None_seq_8192_gbs_256.json │ Generated     │ 7,454 bytes │
└─────────────────────────────────────────────────────────────────────────┴─────────────┴─────────────┘

 CUDA Memory Analysis & Run Status
                                   Memory Usage Analysis & Execution Status                                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃                                          ┃ Memory       ┃              ┃                 ┃                 ┃
┃ Configuration                            ┃ Status       ┃ Run Status   ┃ Est. Usage (GB) ┃ GPU Memory (GB) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ base_config                              │ Safe         │ ▶ Run        │ 19.2            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤
│ gemma2_9b_8nodes_tp_2_pp_2_cp_1_ep_1_mbs… │ Safe         │ ▶ Run        │ 21.9            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤
│ gemma2_9b_8nodes_tp_2_pp_2_cp_1_ep_1_mbs… │ Safe         │ ▶ Run        │ 21.9            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤
│ gemma2_9b_8nodes_tp_2_pp_2_cp_1_ep_1_mbs… │ Safe         │ ▶ Run        │ 27.4            │ 141             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────┼─────────────────┤

Memory Analysis Summary:
Safe configurations (will run): 67
Potential OOM configurations (will be skipped): 0
Performance Results: Not available
```

### 2. Run Training Experiments (`run`)

Executes training experiments using generated configurations.

```bash
python launcher.py run \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model gemma2_9b \
  --launcher-node-group tme-nebius-h200-01 \
  --training-node-group tme-nebius-h200-01 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace \
   --sequential \
   --run-all

### 3. List Configurations (`list-configs`)

Lists available configurations for a model.

```bash
python launcher.py list-configs \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model llama31_70b \
  --launcher-node-group tme-nebius-h200-01 \
  --training-node-group tme-nebius-h200-01 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace
```

### 4. Analyze Results (`results`)

Analyzes training results and generates performance reports.

```bash
python launcher.py results \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model llama31_8b \
  --logs-path /nemo-workspace/autotuner/new/logs/llama31_8b \
  --log-prefix nemo \
  --launcher-node-group tme-nebius-h200-01 \
  --top-n 10 \
  --cost-per-gpu-hour 3.0 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace
```

**Expected output:**
```
Performance & Cost Analysis Summary
================================================================================

Best Performing Configuration: 
llama_8b_8nodes_tp_4_pp_1_cp_1_ep_1_mbs_2_vp_1_seq_8192_gbs_512
  M-TFLOPs/GPU: 1894.61
  Time per Global Step: 1.8100s
  Total Training Time: 74.9 days
  Total Training Cost: $345,230.10

Base Configuration: 
llama_8b_8nodes_tp_1_pp_1_cp_2_ep_1_mbs_1_vp_None_seq_8192_gbs_512
  M-TFLOPs/GPU: 534.98
  Time per Global Step: 6.4100s
  Total Training Time: 265.3 days
  Total Training Cost: $1,222,610.47

 Best vs Base Performance & Cost Savings:
  M-TFLOPs/GPU improvement: +254.1%
  Training time savings: 4569.7 hours (190.4 days)
  Cost savings: $877,380.37 (+71.8%)
   Total Savings: $877,380.37

 Top 5 Configurations - Performance & Cost Analysis
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃   ┃                                                                   ┃          ┃         ┃          ┃            ┃
┃ … ┃ TP/PP/CP/EP/VP                                                    ┃ MBS/GBS   ┃ M-TFLOPS/GPU┃ Days    ┃ Cost     ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 1 │ 4/1/1/1/1                                                         │ 2/512      │ 1894.61  │ 74.9   │ $345,230   │
├──────┼─────────────────────────────────────────────────────────────────┼────────────┼──────────────┼────────┼────────────┤
│ 2 │ 4/1/1/1/1                                                         │ 1/512      │ 1863.72  │ 76.2   │ $350,952   │
├──────┼─────────────────────────────────────────────────────────────────┼────────────┼──────────────┼────────┼────────────┤
│ 3 │ 2/4/2/1/1                                                         │ 4/512      │ 1824.06  │ 77.8   │ $358,582   │
├──────┼─────────────────────────────────────────────────────────────────┼────────────┼──────────────┼────────┼────────────┤
│ 4 │ 4/1/2/1/1                                                         │ 2/512      │ 1814.41  │ 78.2   │ $360,489   │
├──────┼─────────────────────────────────────────────────────────────────┼────────────┼──────────────┼────────┼────────────┤
│ 5 │ 4/1/2/1/1                                                         │ 1/512      │ 1758.58  │ 80.7   │ $371,933   │
└──────┴─────────────────────────────────────────────────────────────────┴────────────┴──────────────┴────────┴────────────┘

 Table Legend
================================================================================
TP/PP/CP/EP/VP: Tensor/Pipeline/Context/Expert/Virtual Parallelism
Seq: Sequence Length
MBS/GBS: Micro Batch Size / Global Batch Size
M-TFLOPS/GPU: Millions of TFLOPS per GPU

 Full Configuration Names (for reference)
================================================================================
1. [Best] llama_8b_8nodes_tp_4_pp_1_cp_1_ep_1_mbs_2_vp_1_seq_8192_gbs_512
2. [Generated] llama_8b_8nodes_tp_4_pp_1_cp_1_ep_1_mbs_1_vp_1_seq_8192_gbs_512
3. [Generated] llama_8b_8nodes_tp_2_pp_4_cp_2_ep_1_mbs_4_vp_1_seq_8192_gbs_512
4. [Generated] llama_8b_8nodes_tp_4_pp_1_cp_2_ep_1_mbs_2_vp_1_seq_8192_gbs_512
5. [Generated] llama_8b_8nodes_tp_4_pp_1_cp_2_ep_1_mbs_1_vp_1_seq_8192_gbs_512

 Recommendations
========================================
Best Performance: 
'llama_8b_8nodes_tp_4_pp_1_cp_1_ep_1_mbs_2_vp_1_seq_8192_gbs_512'
Most Cost-Efficient: 
'llama_8b_8nodes_tp_4_pp_1_cp_1_ep_1_mbs_2_vp_1_seq_8192_gbs_512'
Switch from base config to save: $877,380.37

Cost analysis completed successfully!

### 5. List Models (`list-models`)

Lists all supported models.

```bash
python launcher.py list-models
```

## Complete Workflow Example

Here's a complete workflow for optimizing a Gemma2 9B model on an 8-node H200 cluster:

### Step 1: Generate Configurations
```bash
python launcher.py generate \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model gemma2_9b \
  --launcher-node-group tme-nebius-h200-01 \
  --training-node-group tme-nebius-h200-01 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace \
  --resource-shape gpu.8xh200 \
  --nodes 8 \
  --gpus-per-node 8 \
  --logs-subdir /nemo-workspace/autotuner/new/logs
```

### Step 2: Run Training Experiments
```bash
python launcher.py run \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model gemma2_9b \
  --launcher-node-group tme-nebius-h200-01 \
  --training-node-group tme-nebius-h200-01 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace
```

### Step 3: Analyze Results
```bash
python launcher.py results \
  --config-dir /nemo-workspace/autotuner/new/generated_configs \
  --model gemma2_9b \
  --logs-path /nemo-workspace/autotuner/new/logs/gemma2_9b \
  --log-prefix nemo \
  --launcher-node-group tme-nebius-h200-01 \
  --top-n 10 \
  --cost-per-gpu-hour 3.0 \
  --mount-from node-nfs:lepton-shared-fs \
  --mount-source-path / \
  --mount-path /nemo-workspace
```

## Remote Execution

All operations are executed remotely on DGX Cloud Lepton clusters using NeMo Run:

1. **Script Generation**: Python scripts are dynamically generated and embedded in remote jobs
2. **Remote Execution**: Jobs are launched on Lepton using LeptonExecutor by NeMo Run
3. **Environment Setup**: NeMo is automatically cloned and installed from source in remote containers
4. **Authentication**: Lepton authentication is handled automatically via `lep login` in remote jobs
5. **Log Streaming**: Real-time logs are streamed to your terminal via NeMo Run
6. **Result Retrieval**: Results are available in the mounted workspace

### How It Works

The launcher uses NeMo Run's `Experiment` framework:

- **`run.LeptonExecutor`**: Manages remote container execution and resource allocation
- **`run.Script`**: Embeds Python code directly in remote jobs (no file uploads needed)
- **`run.run()`**: Executes the remote job and streams logs back to your terminal

### One-time Setup

The launcher automatically handles setup tasks in each remote job:

- **NeMo Installation**: Clones your NeMo fork to `/tmp/nemo-source` and installs in editable mode
- **Environment Setup**: Configures Python paths and sets up required environment variables
- **Lepton Authentication**: Runs `lep login -c {workspace_id}:{token}` in remote container
- **Dependency Management**: Installs NeMo from source with all required components

## Configuration Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | NeMo model name (e.g., "llama31_70b", "gemma2_9b") |
| `nodes` | int | Number of compute nodes |
| `gpus_per_node` | int | GPUs per node |
| `resource_shape` | str | GPU type and count (e.g., "gpu.8xh200") |
| `config_dir` | str | Directory to save configurations (required) |
| `logs_subdir` | str | Directory to save training logs |
| `launcher_node_group` | str | Node group for launcher jobs |
| `training_node_group` | str | Node group for training jobs |
| `mount_from` | str | Storage backend (e.g., "node-nfs:lepton-shared-fs") |
| `mount_source_path` | str | Source path for mounting |
| `mount_path` | str | Container mount path |

### Parallelism Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_batch_sizes` | List[int] | [256] | Global batch sizes to test |
| `tensor_parallel_sizes` | List[int] | [2] | Tensor parallelism sizes |
| `pipeline_parallel_sizes` | List[int] | [2] | Pipeline parallelism sizes |
| `context_parallel_sizes` | List[int] | [1] | Context parallelism sizes |
| `expert_parallel_sizes` | List[int] | [1] | Expert parallelism sizes |
| `virtual_pipeline_model_parallel_sizes` | List[int] or None | None | Virtual pipeline parallelism sizes |
| `micro_batch_sizes` | List[int] | [1] | Micro batch sizes |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps_per_run` | int | 50 | Steps per experiment |
| `max_steps` | int | 50 | Total training steps |
| `seq_length` | int | 8192 | Sequence length |
| `num_tokens_in_b` | int | 1000 | Tokens in billions |

### Infrastructure Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `container_image` | str | "nvcr.io/nvidia/nemo:25.07" | Container image (NVIDIA NeMo) |
| `max_model_parallel_size` | int | 64 | Maximum model parallel size |

## Output Files

### Generated Files

- `config_dir/model_name/base_config.json` - Base configuration
- `config_dir/model_name/config_*.json` - Generated configurations
- `config_dir/model_name/args.json` - Arguments and metadata
- `logs_subdir/model_name/` - Training logs for each configuration

### Results Analysis

The `results()` function displays:
- **Performance Summary**: Best/worst/base configurations with detailed metrics
- **Cost Analysis**: Training time and cost comparisons with percentage improvements
- **Top Configurations**: Ranked by performance with compact, readable table
- **Configuration Reference**: Full configuration names for easy identification
- **Cost Efficiency**: Most cost-effective configuration analysis
- **Recommendations**: Best performance vs cost efficiency with savings

## 🐛 Troubleshooting

### Common Issues

1. **Launcher commands not parsing**
   - Check for trailing spaces after backlashes in your command and remove them if any 
   - Check for missing Arguements in any command 

2. **Job Fails Immediately**
   - Check DGX Cloud Lepton authentication: `lep workspace list`
   - Verify environment variables are set: `echo $LEPTON_AUTOTUNER_WORKSPACE_ID $LEPTON_AUTOTUNER_TOKEN`
   - Check workspace mounting: Ensure `/nemo-workspace` is accessible

3. **Configuration Generation Fails**
   - Verify model name is supported: `python launcher.py list-models`
   - Check resource requirements match available resources
   - Ensure config directory is writable

4. **Training Experiments Fail**
   - Check GPU availability
   - Verify data paths and permissions
   - Review logs for specific error messages

5. **Results Analysis Issues**
   - Ensure log files exist and are readable
   - Check log prefix matches actual log files

### Getting Help

```bash
# Show general help
python launcher.py --help

# Show help for specific command
python launcher.py generate --help
python launcher.py run --help
python launcher.py results --help
python launcher.py list-configs --help
```

## Configuration Tips

### Model Selection
- Use exact NeMo model names: `"llama31_70b"`, `"gemma2_9b"`, `"mixtral_8x7b"`, etc.
- Check supported models with: `python launcher.py list-models`

### Implementation Details
- **NeMo Run Integration**: Uses `nemo_run.LeptonExecutor` and `nemo_run.Script` for remote execution
- **Script Generation**: Python code is embedded directly in remote jobs (no file uploads)
- **NeMo Installation**: Automatically clones and installs NeMo from source in remote containers
- **Authentication**: Handles Lepton auth via `lep login` in remote environment

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
- Ensure `config_dir` and `logs_subdir` are absolute paths
- The `args.json` file will be saved in `config_dir/model_name/args.json`

This README provides a comprehensive guide to using the AutoTune module step-by-step, with realistic examples and troubleshooting tips.
