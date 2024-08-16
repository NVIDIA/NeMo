#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=coreai_dlalgo_llm-run:t5_mcore
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=04:00:00

MCORE_VERSION='mcore0_8'

CONT="nvcr.io/nvidia/pytorch:23.08-py3"
MOUNT="/lustre/fsw/:/lustre/fsw/"

### Model's arguments setup
CHECKPOINT_PATH="/lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_mcore/release_tests/results/t5_mcore_sbatch_releasetest_${MCORE_VERSION}"
VOCAB_FILE="/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/Pile_data/Pile/bert-large-cased-vocab.txt"
DATA_PATH="/lustre/fsw/coreai_dlalgo_genai/huvu/data/t5/Pile_data/Pile/my-t5_00_bert_tokenizer_text_document"
TENSORBOARD_DIR=$CHECKPOINT_PATH
LOG_DIR=$CHECKPOINT_PATH

MBS=64
GBS=$(($SLURM_JOB_NUM_NODES*$MBS*8))

# set hyper-params
T5_ARGS="\
    --encoder-num-layers 12 \
    --decoder-num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 64 \
    --global-batch-size 512 \
    --lr 0.0001 \
    --train-iters 1000000 \
    --lr-decay-iters 1000000 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --vocab-extra-ids 100 \
    --init-method-std 0.015 \
    --transformer-impl transformer_engine \
    --use-mcore-models \
"

DATA_ARGS="\
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --tokenizer-type BertWordPieceCase \
    --split 99982,9,9 \
"

OUTPUT_ARGS="\
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --log-interval 100 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-validation-ppl-to-tensorboard \
    --log-timers-to-tensorboard \
    --timing-log-level 2 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --save-interval 5000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --distributed-backend nccl
"

ALL_ARGS="${T5_ARGS} ${DATA_ARGS} ${OUTPUT_ARGS}"
echo $ALL_ARGS

### Running job
mkdir $CHECKPOINT_PATH
OUTFILE=$LOG_DIR/slurm-%j.out
ERRFILE=$LOG_DIR/error-%j.out
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Running training script."
srun -o ${OUTFILE} -e ${ERRFILE} --mpi=pmix \
    --container-image="${CONT}" --container-mounts="${MOUNT}" \
    --no-container-mount-home \
    --ntasks-per-node=8 \
    -N ${SLURM_JOB_NUM_NODES}  \
    bash -c "cd /lustre/fsw/coreai_dlalgo_genai/huvu/codes/T5_mcore/release_tests/${MCORE_VERSION}/megatron-lm; \
            pip install -e .; \
            pip install pydantic==2.2.1; \
            python pretrain_t5.py ${ALL_ARGS}"