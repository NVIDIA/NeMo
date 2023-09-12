NEMO=/home/scratch.guyueh_sw/2023su/cpu_offload/NeMo
# MLM=/home/scratch.guyueh_sw/2023su/cpu_offload/megatron-lm
# FLASH_ATTN=/home/scratch.guyueh_sw/2023su/cpu_offload/flash-attention

export PYTHONPATH=${NEMO}:${PYTHONPATH}
# export PYTHONPATH=${NEMO}:${MLM}:${FLASH_ATTN}:${PYTHONPATH}

MICRO_BATCH_SIZE=${1:-1}
SEQ_LENGTH=${2:-512}
MEGATRON_AMP_O2=${3:-"True"}
CHECKPOINT_GRANULARITY=${4:-"full"}
CHECKPOINT_METHOD=${5:-"uniform"}
CHECKPOINT_NUM_LAYERS=${6:-1}

python ${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path ${NEMO}/debug_cpu_offload/megatron \
--config-name gpt_5b.yaml \
trainer.devices=1 \
trainer.num_nodes=1 \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.global_batch_size=${MICRO_BATCH_SIZE} \
model.data.data_impl="mock" model.data.data_prefix=[] \
model.optim.name="fused_adam" \
model.megatron_amp_O2=${MEGATRON_AMP_O2} \
model.encoder_seq_length=${SEQ_LENGTH} \
model.activations_checkpoint_granularity=${CHECKPOINT_GRANULARITY} \
model.activations_checkpoint_method=${CHECKPOINT_METHOD} \
model.activations_checkpoint_num_layers=${CHECKPOINT_NUM_LAYERS} \
2>&1 | tee gpt_5b_recompute_MBS_${MICRO_BATCH_SIZE}_seq_${SEQ_LENGTH}_amp_O2_${MEGATRON_AMP_O2}_granularity_${CHECKPOINT_GRANULARITY}_method_${CHECKPOINT_METHOD}_num_layers_${CHECKPOINT_NUM_LAYERS}.log