NEMO=/home/scratch.guyueh_sw/2023su/cpu_offload/NeMo
# MLM=/home/scratch.guyueh_sw/2023su/cpu_offload/megatron-lm
# FLASH_ATTN=/home/scratch.guyueh_sw/2023su/cpu_offload/flash-attention

export PYTHONPATH=${NEMO}:${PYTHONPATH}
# export PYTHONPATH=${NEMO}:${MLM}:${FLASH_ATTN}:${PYTHONPATH}

MICRO_BATCH_SIZE=${1:-1}
SEQ_LENGTH=${2:-512}
MEGATRON_AMP_O2=${3:-"True"}

python ${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path ${NEMO}/debug_cpu_offload/megatron \
--config-name gpt_3.4b_no_grad_acc_fusion.yaml \
trainer.devices=1 \
trainer.num_nodes=1 \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.global_batch_size=16 \
model.data.data_impl="mock" model.data.data_prefix=[] \
model.optim.name="fused_adam" \
model.megatron_amp_O2=${MEGATRON_AMP_O2} \
model.encoder_seq_length=${SEQ_LENGTH} \
2>&1 | tee gpt_3.4b_MBS_${MICRO_BATCH_SIZE}_seq_${SEQ_LENGTH}_amp_O2_${MEGATRON_AMP_O2}.log