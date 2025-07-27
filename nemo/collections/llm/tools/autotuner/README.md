# AutoTuner

AutoTuner is a fully automated, predictive configuration builder and orchestrated tool that achieves maximum model throughput for LLM Training.

## Overview

AutoTuner extends NeMo's Auto Configurator with a fully automated, production-ready pipeline that searches for hyperparameters achieving maximum training throughput while automatically preventing CUDA OOM errors and providing comprehensive performance & cost analysis for Large Language Models. Unlike research-only tools, AutoTuner runs directly on your GPU infrastructure, making it ready for production deployment and real-world training optimization.

### Note

AutoTuner is supported for all NeMo models including GPT-based models: GPT3, LLama, Mixtral, Mistral, Gemma, Nemotron, Starcoder, Qwen, and more.

### AutoTuner Capabilities

AutoTuner is designed to iterate over different model configurations quickly and find the best configuration that maximizes training throughput while minimizing both time and money. It offers comprehensive features to facilitate this, as detailed below.

**Enhanced Auto Configurator Features:**
- **Model size recommendation**: Finds the optimal model size if the parameter is not specified
- **Training time estimation**: Estimates model training time based on input parameters
- **Hyperparameters recommendation**: Finds the optimal list of hyperparameters to be trained
- **Optimal configuration recommendation**: Calculates the performance after a short training of candidate configurations and finds the optimal model configuration

**Production-Ready AutoTuner Features:**
- **CUDA OOM Prevention**: Automatically detects and filters out configurations that would cause out-of-memory errors, preventing training failures
- **Orchestrated Pipeline**: Fully automated workflow from configuration generation to results analysis with no manual intervention required
- **Rigorous Performance Analysis**: Comprehensive metrics including TFLOPS, training time, samples per second, and cost optimization
- **Cost Analysis**: Detailed cost breakdown per configuration including training time, GPU hours, and total expenditure
- **Memory Analysis**: Conservative memory estimation with safety margins to ensure reliable training
- **Rich Visualization**: Beautiful tables and charts for easy decision making and result interpretation
- **Production Deployment**: Runs directly on customer GPU infrastructure, ready for enterprise use

## Installation

```bash
# Ensure you have NeMo, nemo_run and rich installed
pip install nemo nemo_run rich
```

## Ready-to-Use Scripts

The `autotuner_scripts/` directory contains three ready-to-use scripts that demonstrate the complete AutoTuner workflow:

### Step 1: Generate Configurations

**File: `autotuner_scripts/generate_configs.py`**  
Generate and validate training configurations for your model (including predictive OOM analysis):

```python
from nemo.collections.llm.tools.autotuner import generate, list_configs
from nemo.collections.llm.tools.autotuner import AutoTuneArgs

# Create AutoTuneArgs object with all parameters
args = AutoTuneArgs(
    # Required parameters
    model="gemma2_2b",                         # NeMo model name
    nodes=1,                                # Number of compute nodes
    gpus_per_node=8,                        # Number of GPUs per node
    resource_shape="gpu.8xh200",            # GPU resource shape
    
    # Optional parallelism settings
    tensor_parallel_sizes=[1, 2, 4],     # Tensor parallelism sizes to test
    pipeline_parallel_sizes=[1, 2],         # Pipeline parallelism sizes to test
    context_parallel_sizes=[1, 2],       # Context parallelism sizes to test
    expert_parallel_sizes=[1],        # Expert parallelism sizes to test
    
    # Optional batch size settings
    global_batch_sizes=[256, 512],    # Global batch sizes to test
    micro_batch_sizes=[1, 2, 4],         # Micro batch sizes to test
    
    # Optional training parameters
    max_steps_per_run=50,                   # Training steps per experiment
    seq_length=8192,                        # Sequence length for training
    num_tokens_in_b=15000,                  # Total tokens in billions
    
    # Optional infrastructure settings
    container_image="nvcr.io/nvidia/nemo:25.04",  # Container image
    mount_path="/nemo-workspace",           # Mount path for containers
    mount_from="node-nfs:shared",           # Storage backend
    node_group="nebius-h200-01",            # Node group name
    
    # Optional output settings
    config_dir="/path/to/your/configs",       # Directory to save configurations
    logs_subdir="/path/to/your/logs"          # Directory to save logs
)

# Generate configurations using the args object
result = generate(args)
list_configs(args.config_dir, args.model)
```

**Usage:**
```bash
cd autotuner_scripts
python3 generate_configs.py
```

**What it does:**
- Creates an `AutoTuneArgs` object with your model parameters
- Generates multiple configurations with different parallelization strategies
- Performs memory analysis to identify potential OOM configurations
- Saves configurations to `config_dir/model_name/`
- Displays a summary table of all generated configurations

### Step 2: Run Training Experiments

**File: `autotuner_scripts/run_experiments.py`**  
Execute training experiments for all generated configurations:

```python
from nemo.collections.llm.tools.autotuner import run
from nemo.collections.llm.tools.autotuner import AutoTuneArgs

# Load configuration from the generated args.json file
args = AutoTuneArgs.load_from_file("/path/to/your/configs/model_name/args.json")

# Run all configurations using the loaded args
results = run(args)

print(f"Completed {len(results)} experiments")
```

**Usage:**
```bash
cd autotuner_scripts
python3 run_experiments.py
```

**What it does:**
- Loads the `AutoTuneArgs` from the generated `args.json` file
- Executes training runs for all configurations
- Skips configurations flagged as potential OOM risks
- Saves training logs and performance metrics

### Step 3: Analyze Results

**File: `autotuner_scripts/results.py`**  
Analyze training results and provide detailed performance insights:

```python
from nemo.collections.llm.tools.autotuner import results
from nemo.collections.llm.tools.autotuner import AutoTuneArgs

# Load configuration from the generated args.json file
args = AutoTuneArgs.load_from_file("/path/to/your/configs/model_name/args.json")

# Get detailed results analysis using the loaded args
analysis = results(
    args=args,                                    # AutoTuneArgs object
    logs_path="/path/to/your/logs/model_name",   # Path to training logs directory
    log_prefix="nemo",                           # Log file prefix to search for
    top_n=5,                                     # Number of top configurations to display
    force_reconstruct=False,                     # Force reconstruction of config objects
    cost_per_node_hour=3.0,                      # Cost per node per hour for cost analysis
    quiet=False                                  # Suppress rich console output
)
```

**Usage:**
```bash
cd autotuner_scripts
python3 results.py
```

**What it does:**
- Loads training results from logs
- Calculates performance metrics for each configuration
- Provides cost analysis and recommendations
- Shows top-performing configurations

## Complete Workflow Example

Here's how to use all three scripts in sequence:

### Step 1: Generate Configurations
```bash
# Edit generate_configs.py with your model and parameters
cd autotuner_scripts
python3 generate_configs.py
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

### Step 2: Run Training Experiments
```bash
# Edit run_experiments.py with correct paths
python3 run_experiments.py
```


### Step 3: Analyze Results
```bash
# Edit results.py with correct paths
python3 results.py
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
   - Check `config_dir` and `logs_subdir` in `generate_configs.py`
   - Update `args.json` path in `run_experiments.py` and `results.py`
   - Ensure paths are absolute or relative to script location

3. **Model Not Found**: Check `list_models()` for supported models
   ```python
   from nemo.collections.llm.tools.autotuner import list_models
   list_models()
   ```

4. **Configuration Validation Failed**: Review parallelism constraints
   - Ensure `nodes * gpus_per_node` matches total GPU count
   - Parallelism sizes must divide evenly into total GPU count

5. **No Configurations to Run**: All configs flagged for OOM
   - Reduce `micro_batch_sizes` or increase parallelism
   - Use `run_all=True` to force run all configurations (not recommended)

6. **Logs Directory Not Found**: Run training experiments first
   - Ensure `run_experiments.py` completes successfully before running `results.py`

### Memory Issues

- Reduce `micro_batch_sizes` or increase parallelism
- Use `run_all=True` to run OOM-risk configurations anyway
- Check GPU memory specifications in `resource_shape`

### Performance Issues

- Increase `max_steps_per_run` for more accurate timing
- Use `force_reconstruct=True` to regenerate configurations
- Check log files for training errors

### Debug Mode

Add debug logging to any script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Configuration Inspection

Inspect generated configurations:
```python
from nemo.collections.llm.tools.autotuner import list_configs
list_configs("/path/to/configs", "your_model_name")
```

This README provides a comprehensive guide to using the AutoTune module step-by-step, with examples and troubleshooting tips.
