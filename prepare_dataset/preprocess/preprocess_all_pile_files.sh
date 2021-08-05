#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=big_nlp-gpt3:preprocess_all_pile_files
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH -o log-preprocess-%j_%a.out

cd $SLURM_SUBMIT_DIR
echo "Job ID: $SLURM_JOB_ID"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Node ID: $SLURM_NODEID"

srun --container-image nvcr.io#nvidia/pytorch:21.06-py3 --container-mounts $SLURM_SUBMIT_DIR/..:/workspace/bignlp_scripts bash /workspace/bignlp_scripts/prepare_dataset/preprocess/preprocess_single_pile_file.sh &

wait
