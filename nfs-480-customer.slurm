#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/sft_script/results/log-nemo-megatron-Qwen2.5-MoE-200B_%j.err
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nemo-reproduce
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/sft_script/results/log-nemo-megatron-Qwen2.5-MoE-200B_%j.out
#SBATCH --partition=batch

#####################Below are the VARIABLES you may modify##########################

MEGATRON_PATH="/lustre/fsw/portfolios/coreai/users/zhiyul/Megatron-LM"
NEMO_PATH="/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo"

PATH_CONTAINING_TRAINING_VALIDATION_TEST_JSONL="/lustre/fsw/portfolios/coreai/users/zhiyul/data/large_test"
DATA_PATH=${PATH_CONTAINING_TRAINING_VALIDATION_TEST_JSONL}
MICRO_BATCHSIZE=1
STEPS=100
PRECISION="bf16"

# MODEL="Qwen/Qwen3-30B-A3B"
MODEL="Qwen/Qwen3-235B-A22B"

if [ $MODEL == "Qwen/Qwen3-235B-A22B" ]; then
    MODEL_CKPT="/lustre/fsw/portfolios/coreai/users/zhiyul/nemo/Qwen/Qwen3-235B-A22B/"
    NUM_NODES=16
    DEVICES_PER_NODE=8
    # $TP * $CP * $PP * $DP == 8
    TP=4
    CP=1
    PP=16
    EP=4
    ETP=1
    #VPP=2
    # PP_LAYOUT="E\(t\)\*10\|\(tttttttttttt\|\)\*6,ttttttttttttL"
    GLOBAL_BATCHSIZE=256
    SEQUENCE_LEN=4096
elif [ $MODEL == "Qwen/Qwen3-30B-A3B" ]; then
    MODEL_CKPT="/lustre/fsw/portfolios/coreai/users/zhiyul/nemo/Qwen/Qwen3-30B-A3B/"
    NUM_NODES=2
    DEVICES_PER_NODE=8
    TP=1
    CP=1
    PP=2
    EP=8
    ETP=1
    GLOBAL_BATCHSIZE=16
    SEQUENCE_LEN=4096
else
    echo "Model not supported"
    exit 1
fi


NAME="nemo-dev-zhiyul"
LOG_DIR="${NEMO_PATH}/sft_script/finetune_logs/"

SEQ_PACK_LEN=${SEQUENCE_LEN}

USE_FLASH_ATTN=false
NSYS_PROFILE=false

#####################################################################################
if [[ $USE_FLASH_ATTN != true ]]; then
    flash_option=""
else
    echo "Use flash backend for attn"
    flash_option=" \
    --use-flash-attn"
fi

if [[ -z "$ETP" ]]; then
    etp_option=""
else
    etp_option=" \
    --etp_size $ETP"
fi

if [[ -z "$VPP" ]]; then
    vpp_option=""
else
    vpp_option=" \
    --vpp_size $VPP"
fi

if [[ -z "$PP_LAYOUT" ]]; then
    pp_layout_option=""
else
    pp_layout_option=" \
    --pp_layout $PP_LAYOUT"
fi

if [[ -z "$SEQ_PACK_LEN" ]]; then
    pack_option=""
else
    pack_option=" \
    --packing_length $SEQ_PACK_LEN"
fi

if [ $NSYS_PROFILE != true ]; then
    launch_cmd="python3"
    nsys_option=""
else
    echo "Launch script with nsys profling"
    launch_cmd="nsys profile \
            -s none \
            -t cuda,nvtx \
            -o ${NEMO_PATH}/sft_script/profile_logs/profile_\${SLURM_JOB_ID}_node\${SLURM_NODEID}_rank\${SLURM_PROCID} \
            --force-overwrite true \
            --cuda-memory-usage true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            python3"
    nsys_option="--nsys-profile"
fi

# Function to set NCCL and other environment variables
with_nccl_local_env() {
    export UCX_IB_PCI_RELAXED_ORDERING=1
    export NCCL_IB_PCI_RELAXED_ORDERING=1
    export NCCL_IB_TIMEOUT=22
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10000
    export NCCL_DEBUG=WARN
    export TRANSFORMERS_OFFLINE=1
    export TORCH_NCCL_AVOID_RECORD_STREAMS=1
    export NCCL_NVLS_ENABLE=0
    export NCCL_IB_DISABLE=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVIDIA_PYTORCH_VERSION=24.08
    export NEMO_LOG_MEMORY_USAGE=1
    export TOKENIZERS_PARALLELISM=False
    if [[ -z "$MEGATRON_PATH" ]]; then
        export PYTHONPATH=${NEMO_PATH}
    else
        export PYTHONPATH=${MEGATRON_PATH}:${NEMO_PATH}
    fi
    
    # Flash attention exports (if enabled)
    if [[ $USE_FLASH_ATTN == true ]]; then
        export NVTE_FUSED_ATTN=0
        export NVTE_FLASH_ATTN=1
    fi
    
    # Execute the command with the environment set
    exec "$@"
}

CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/nemo-25.07.sqsh"
MOUNTS="/lustre:/lustre"
srun \
    --export=ALL,HOME=/tmp \
    --no-container-mount-home \
    --container-mounts ${MOUNTS} \
    --container-image=${CONTAINER_NAME} \
    --container-workdir /lustre/fsw/portfolios/coreai/users/zhiyul/NeMo \
    --pty bash -c "
source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
sleep 60
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $(with_nccl_local_env) ${launch_cmd} ${NEMO_PATH}/sft_script/gpt_sft.py \
    --name ${NAME} \
    --model_path ${MODEL_CKPT} \
    --tp_size ${TP} \
    --cp_size ${CP} \
    --pp_size ${PP} \
    --ep_size ${EP} \
    --precision ${PRECISION} \
    --devices ${DEVICES_PER_NODE} \
    --num_nodes ${NUM_NODES} \
    --log_dir ${LOG_DIR} \
    --max_steps ${STEPS} \
    --gbs ${GLOBAL_BATCHSIZE} \
    --mbs ${MICRO_BATCHSIZE} \
    --data_path ${DATA_PATH} \
    --seq_length ${SEQUENCE_LEN} \
    --lr 4e-5 \
    --min_lr 4e-6 \
    --warmup_steps 109 \
    --log_interval 1 \
    --val_check_interval 5 \
    --recompute_type selective \
    --use-hf-chat-template \
    ${flash_option} \
    ${etp_option} \
    ${vpp_option} \
    ${pp_layout_option} \
    ${pack_option} \
    ${nsys_option}
"