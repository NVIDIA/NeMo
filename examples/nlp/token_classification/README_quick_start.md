# NeMo Quick Start - Token Classification

This is a quick start guide for testing NeMo with token classification (Named Entity Recognition) that should complete within 30 minutes.

## Prerequisites

1. Install NeMo and its dependencies:
```bash
pip install nemo_toolkit[nlp]
```

2. Install PyTorch (if not already installed):
```bash
pip install torch
```

## Running the Quick Start

### Local Run

1. Navigate to the token classification directory:
```bash
cd examples/nlp/token_classification
```

2. Run the quick start script:
```bash
python quick_start.py
```

### Running in Slurm Environment

1. Navigate to the token classification directory:
```bash
cd examples/nlp/token_classification
```

2. Edit the `sbatch.sh` script:
   - Update the partition name (`--partition=your_partition`)
   - Adjust memory requirements if needed (`--mem=32G`)
   - Modify module loads based on your cluster setup

3. Submit the job:
```bash
sbatch sbatch.sh
```

4. Monitor the job:
```bash
squeue -u $USER
```

5. Check the logs:
```bash
cat nemo_quick_test_<job_id>.log
```

The script uses a separate configuration file (`conf/quick_start_config.yaml`) that is optimized for quick testing. This keeps the original configuration intact while providing a fast way to test NeMo functionality.

## What to Expect

- Training will use a small BERT model (2 layers instead of 12)
- Uses the CoNLL-2003 dataset for Named Entity Recognition
- Training will run for 2 epochs
- Uses mixed precision (FP16) for faster training
- Includes OneLogger telemetry integration
- Should complete within 30 minutes on a single GPU

## Configuration Details

The quick start uses the following optimizations:
- Small batch size (16) to ensure it runs on any GPU
- Reduced model size (2 layers instead of 12)
- Mixed precision training
- Cached dataset loading
- Minimal logging and checkpointing

## Monitoring

You can monitor the training progress through:
1. Console output (logs every 10 steps)
2. OneLogger telemetry (if available)
3. Training metrics (loss, accuracy)
4. Slurm job status and logs

## Troubleshooting

If you encounter any issues:
1. Check GPU memory usage
2. Verify PyTorch and NeMo versions
3. Ensure you have enough disk space for the dataset
4. Check the logs for any error messages
5. Verify Slurm resource allocation
6. Check module availability on your cluster

## Configuration Files

- `conf/quick_start_config.yaml`: Optimized for quick testing (30 minutes)
- `conf/token_classification_config.yaml`: Original configuration for full training
- `sbatch.sh`: Slurm batch script for cluster execution 