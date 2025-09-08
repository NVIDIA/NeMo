#!/bin/bash

# Configuration variables
NUM_PROC_PER_NODE=4  # Set the number of processes per node
DATA_TYPE="energon"
DATA_PATH="/datasets/wds"

LANGUAGE_MODEL_PATH="/models/Qwen/Qwen2-VL-2B-Instruct"

# Parallelism configruration
DEVICES=4
TP_SIZE=1
# CP_SIZE=2

# Batch size
MBS=1
GBS=4

#Pixel Nums
MINPIXELS=784
MAXPIXELS=43904

# Exp logging path
EXPERIMENT_NAME="Qwen2VL_finetune_2B${TP_SIZE}_CP${CP_SIZE}_MBS${MBS}_GBS${GBS}_seqpack"
WANDB_PROJECT="Qwen2VL"
LOG_DIR="./experiments_finetune"
# RESTORE_PATH="/workspace/experiments_pretrain/LLaVA-NeXT_ptr_TP4_CP1_seqpack_with_val_loss--reduced_train_loss=2.0088-epoch=46-consumed_samples=44800.0"

# Construct the arguments stringring
ARGS="--data_type $DATA_TYPE \
      --data_path $DATA_PATH \
      --image_folder /datasets/ \
      --devices $DEVICES \
      --tp_size $TP_SIZE \
      --name $EXPERIMENT_NAME \
      --mbs $MBS \
      --gbs $GBS \
      --wandb_project $WANDB_PROJECT \
      --log_dir $LOG_DIR\
      --max_sequence_length 4096\
      --min_pixels $MINPIXELS\
      --max_pixels $MAXPIXELS\
      --restore_path $LANGUAGE_MODEL_PATH "


# Run the experiment with torchrun
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "ARGS: $ARGS"
echo "NUM_PROC_PER_NODE: ${NUM_PROC_PER_NODE}"
echo $ARGS

torchrun --nproc_per_node=$NUM_PROC_PER_NODE \
    /workspace/scripts/vlm/qwen2vl_finetune.py $ARGS
