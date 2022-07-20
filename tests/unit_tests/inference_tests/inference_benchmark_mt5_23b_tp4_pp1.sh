#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p luna
#SBATCH -A joc
#SBATCH --job-name=joc-bignlp_inference:inference_benchmark_mt5_23b_tp4_pp1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --time=0:30:00

srun --mpi=pmix --container-image gitlab-master.nvidia.com#dl/dgx/bignlp/infer:infer_update-py3-base --container-mounts /lustre/fsw/joc/donghyukc/bignlp-scripts:/lustre/fsw/joc/donghyukc/bignlp-scripts,/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_mt5_23b:/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_mt5_23b -o /lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_mt5_23b/inference_benchmark_mt5_23b_tp4_pp1-%j.log -e /lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_mt5_23b/inference_benchmark_mt5_23b_tp4_pp1-%j.error  sh -c ' if [ $PMIX_RANK = 0 ] && [ "$PMIX_HOSTNAME" = "$SLURMD_NODENAME" ]; then CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/tritonserver/bin/tritonserver \
--model-repository=/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_mt5_23b/triton & \
sleep 100 && \
bash /lustre/fsw/joc/donghyukc/bignlp-scripts/bignlp/inference_scripts/benchmark_sweep.sh \
/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_mt5_23b \
60 \
20 \
sequence_length \
max_output_len \
250112 \
4 \
1 \
1; \
else CUDA_VISIBLE_DEVICES=0,1,2,3 \
/opt/tritonserver/bin/tritonserver \
--model-repository=/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_mt5_23b/triton; fi'

set +x
