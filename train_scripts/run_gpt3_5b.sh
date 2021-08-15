#!/bin/bash

#SBATCH -t 4:00:00 --nodes=20 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=big_nlp-gpt3:5b-pile-all --comment=metrics

NAME="5b-pile"

DIR="/bignlp-scripts"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $PWD/logs

CHECKPOINT_DIR="${PWD}/checkpoints/${NAME}"
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tensorboard"
mkdir -p ${TENSORBOARD_DIR}
CHECKPOINT_CONTAINER_DIR="${DIR}/train_scripts/checkpoints/${NAME}"
TENSORBOARD_CONTAINER_DIR="${CHECKPOINT_CONTAINER_DIR}/tensorboard"

# Get the data blend
. "${PWD}/../prepare_dataset/gpt3_blend.sh"

BPE_DIR="${DIR}/prepare_dataset/bpe"

options=" \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 4 \
    --global-batch-size 1280 \
    --train-samples 192000000 \
    --lr-decay-samples 166400000 \
    --lr-warmup-samples 244141 \
    --lr 1.2e-4 \
    --min-lr 1.2e-5 \
    --lr-decay-style cosine \
    --log-timers-to-tensorboard \
    --log-interval 100 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file ${BPE_DIR}/vocab.json \
    --merge-file ${BPE_DIR}/merges.txt \
    --save-interval 10000 \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 9999,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --fp16 \
    --DDP-impl torch \
    --tensorboard-dir ${TENSORBOARD_DIR} "

run_cmd="${DIR}/train_scripts/bind.sh --cpu=${DIR}/train_scripts/dgxa100_ccx.sh --mem=${DIR}/train_scripts/dgxa100_ccx.sh python -u ${DIR}/megatron-lm/pretrain_gpt.py ${options}"

srun -l \
     --container-image "nvcr.io#nvidia/pytorch:20.12-py3" \
     --container-mounts "${PWD}/..:${DIR}" \
     --output=$PWD/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

