#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p luna
#SBATCH -A joc
#SBATCH --job-name=joc-bignlp_inference:inference_benchmark_t5_41b_tp8_pp1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --time=0:30:00

srun --mpi=pmix --container-image gitlab-master.nvidia.com#dl/dgx/bignlp/infer:infer_update-py3-base --container-mounts /lustre/fsw/joc/donghyukc/bignlp-scripts:/lustre/fsw/joc/donghyukc/bignlp-scripts,/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_t5_41b:/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_t5_41b -o /lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_t5_41b/inference_benchmark_t5_41b_tp8_pp1-%j.log -e /lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_t5_41b/inference_benchmark_t5_41b_tp8_pp1-%j.error  sh -c ' if [ $PMIX_RANK = 0 ] && [ "$PMIX_HOSTNAME" = "$SLURMD_NODENAME" ]; then CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/opt/tritonserver/bin/tritonserver \
--model-repository=/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_t5_41b/triton & \
sleep 200 && \
bash /lustre/fsw/joc/donghyukc/bignlp-scripts/bignlp/inference_scripts/benchmark_sweep.sh \
/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_t5_41b \
60 \
20 \
sequence_length \
max_output_len \
29184 \
8 \
1 \
1 2 4 8 16 32 64 128 256; \
else CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/opt/tritonserver/bin/tritonserver \
--model-repository=/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_t5_41b/triton; fi'

set +x
