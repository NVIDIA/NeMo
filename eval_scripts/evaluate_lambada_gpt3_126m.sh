#!/bin/bash

#SBATCH -t 0:30:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=joc-gpt-eval:gpt3-126m-lambada

DIR="/bignlp-scripts"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $PWD/logs_eval

NAME="126m-eval-lambada"
TASK="lambada"
mkdir -p $PWD/logs_eval/$NAME

# Make sure checkpoint ends with .nemo
CHECKPOINT_NAME="126m/???.nemo"
CHECKPOINT="${DIR}/train_scripts/checkpoints/${CHECKPOINT_NAME}"

eval_options=" \
               --name=${NAME} \
               --model nemo-gpt3 \
               --tasks ${TASK}  \
               --batch_size 1 \
               --model_args nemo_model=${CHECKPOINT},tensor_model_parallel_size=1 \
               --output_path $PWD/logs_eval/$NAME
"

run_cmd="${DIR}/train_scripts/bind.sh --cpu=${DIR}/train_scripts/dgxa100_ccx.sh --mem=${DIR}/train_scripts/dgxa100_ccx.sh python -u ${DIR}/eval_scripts/eval-harness/main.py $@ ${eval_options}"

srun -l \
     --container-image "nvcr.io#nvidia/pytorch:20.12-py3" \
     --container-mounts "${PWD}/..:${DIR}" \
     --output=$PWD/logs_eval/${name}_%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

