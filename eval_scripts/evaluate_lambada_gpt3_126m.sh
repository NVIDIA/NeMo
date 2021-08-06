#!/bin/bash

#SBATCH -t 0:30:00 --nodes=1 --exclusive --mem=0 --overcommit --ntasks-per-node=8 --dependency=singleton --job-name=joc-gpt-eval:gpt3-126m-lambada

DIR="/bignlp-scripts"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $PWD/logs_eval

name="126m-eval-lambada"
TASK="LAMBADA"

TEST_DATA="/bignlp-scripts/prepare_dataset/test_data/lambada_test.jsonl"
BPE_DIR="${DIR}/prepare_dataset/bpe"
VOCAB_FILE=${BPE_DIR}/vocab.json
MERGE_FILE=${BPE_DIR}/merges.txt

CHECKPOINT_NAME="126m"
CHECKPOINT_CONTAINER_DIR="${DIR}/train_scripts/checkpoints/${CHECKPOINT_NAME}"

eval_options=" \
               --task $TASK \
               --valid-data $TEST_DATA \
               --tokenizer-type GPT2BPETokenizer \
               --strict-lambada \
               --vocab-file $VOCAB_FILE \
               --merge-file $MERGE_FILE \
               --load $CHECKPOINT_CONTAINER_DIR \
               --tensor-model-parallel-size 1 \
		       --pipeline-model-parallel-size 1 \
               --num-layers 12 \
               --hidden-size 768 \
               --num-attention-heads 12 \
               --micro-batch-size 4 \
               --checkpoint-activations \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --log-interval 10 \
               --fp16 \
               --no-load-optim \
               --no-load-rng"

run_cmd="${DIR}/train_scripts/bind.sh --cpu=${DIR}/train_scripts/dgxa100_ccx.sh --mem=${DIR}/train_scripts/dgxa100_ccx.sh python -u ${DIR}/megatron-lm/tasks/main.py $@ ${eval_options}"

srun -l \
     --container-image "nvcr.io#nvidia/pytorch:20.12-py3" \
     --container-mounts "${PWD}/..:${DIR}" \
     --output=$PWD/logs_eval/${name}_%x_%j_$DATETIME.log sh -c "${run_cmd}"

set +x

