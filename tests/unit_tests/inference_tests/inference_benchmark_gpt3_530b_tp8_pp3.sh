#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH -p luna
#SBATCH -A joc
#SBATCH --job-name=joc-bignlp_inference:inference_benchmark_gpt3_530B_tp8_pp3
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --time=0:30:00

srun --mpi=pmix --container-image gitlab-master.nvidia.com#dl/dgx/bignlp/infer:infer_update-py3-base --container-mounts /lustre/fsw/joc/donghyukc/bignlp-scripts:/lustre/fsw/joc/donghyukc/bignlp-scripts,/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_gpt3_530B:/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_gpt3_530B -o /lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_gpt3_530B/inference_benchmark_gpt3_530B_tp8_pp3-%j.log -e /lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_gpt3_530B/inference_benchmark_gpt3_530B_tp8_pp3-%j.error  sh -c ' if [ $PMIX_RANK = 0 ] && [ "$PMIX_HOSTNAME" = "$SLURMD_NODENAME" ]; then CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/opt/tritonserver/bin/tritonserver \
--model-repository=/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_gpt3_530B/triton & \
sleep 300 && \
bash /lustre/fsw/joc/donghyukc/bignlp-scripts/bignlp/inference_scripts/benchmark_sweep.sh \
/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_gpt3_530B \
60 \
20 \
input_lengths \
request_output_len \
51200 \
8 \
3 \
1 2 4 8 16 32 64 128 256; \
else CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/opt/tritonserver/bin/tritonserver \
--model-repository=/lustre/fsw/joc/donghyukc/bignlp-scripts/results/infer_benchmark_gpt3_530B/triton; fi'

set +x
