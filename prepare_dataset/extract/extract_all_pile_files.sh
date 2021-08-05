#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=big_nlp-gpt3:extract_all_pile_files
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH -o log-extract-%j_%a.out

cd $SLURM_SUBMIT_DIR
echo "Job ID: $SLURM_JOB_ID"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Node ID: $SLURM_NODEID"

srun bash extract/extract_single_pile_file.sh &

wait
