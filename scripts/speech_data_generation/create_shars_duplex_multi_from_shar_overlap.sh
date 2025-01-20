#!/bin/bash
#SBATCH -A llmservice_nemo_speechlm
#SBATCH -N 1 # number of nodes
#SBATCH -t 1:30:00              # wall time
#SBATCH --time-min 01:30:00  
#SBATCH --ntasks-per-node=1    # n tasks per machine (one task per gpu) <required>
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH -J "llmservice_nemo_speechlm-speechllm:canary_v0_speechllm"            # job name (<< CHANGE ! >>)
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# for i in `seq 1 1 511`; do  autorun -n 1 "/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex3/scripts/speech_data_generation/create_shars_duplex_multi_from_shar.sh $i"; done
set -x
CONTAINER=/lustre/fsw/portfolios/llmservice/users/zhehuaic/containers/nemo_s2s_24.08zhc.sqsh
MOUNTS="--container-mounts=/lustre/:/lustre/,/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_stream_dbg2/:/code,/lustre/fsw/portfolios/llmservice/users/zhehuaic/results/:/results,/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data:/data,/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12,/lustre/fsw/portfolios/llmservice/users/zhehuaic/results/HFCACHE/:/hfcache/"
logdir=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/overlap_Mixtral8x22b_MMLPC_en/logdir
mkdir -p $logdir
OUTFILE=${logdir}/slurm-%j-%n.out
ERRFILE=${logdir}/error-%j-%n.out
i=$1

cmd="
python  /lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/code/NeMo_s2s_duplex3/scripts/speech_data_generation/create_shars_duplex_multi_from_shar_overlap.py --in_dir /lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM/SpeechQA/Mixtral8x22b_MMLPC_en/onfly_timestamp_s2s_shars/manifest_${i}_answer/ --manifest /lustre/fsw/portfolios/edgeai/projects/edgeai_riva_rivamlops/data/ALM/SpeechQA/Mixtral8x22b_MMLPC_en/original_manifests/manifest_${i}.json --in_dir_question /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/s2s_synthetic_data/Mixtral8x22b_MMLPC_en/question_shars/manifest_${i}_answer/  --out_shar_dir /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/overlap_Mixtral8x22b_MMLPC_en/manifest_${i}/  --num_shard 10 --turn_silence_sec 0.64 \
  "
srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"

