NEMO=/home/scratch.guyueh_sw/2023su/cpu_offload/NeMo
MLM=/home/scratch.guyueh_sw/2023su/cpu_offload/megatron-lm
FLASH_ATTN=/home/scratch.guyueh_sw/2023su/cpu_offload/flash-attention

export PYTHONPATH=${NEMO}:${MLM}:${FLASH_ATTN}:${PYTHONPATH}

MICRO_BATCH_SIZE=${1:-1}
SEQ_LENGTH=${2:-512}
MEGATRON_AMP_O2=${3:-"True"}
OFFLOAD_REGION=${4:-"encoder"}
OFFLOAD_NUM_LAYERS=${5:-15}

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -o ./nsys_gpt_5b_offload_MBS_${MICRO_BATCH_SIZE}_seq_${SEQ_LENGTH}_amp_O2_${MEGATRON_AMP_O2}_offload_num_layers_${OFFLOAD_NUM_LAYERS}_region_${OFFLOAD_REGION} \
-t cuda,nvtx --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python ${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path ${NEMO}/debug_cpu_offload/megatron \
--config-name gpt_5b.yaml \
trainer.devices=1 \
trainer.num_nodes=1 \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.global_batch_size=4 \
model.data.data_impl="mock" model.data.data_prefix=[] \
model.optim.name="fused_adam" \
model.megatron_amp_O2=${MEGATRON_AMP_O2} \
model.encoder_seq_length=${SEQ_LENGTH} \
++model.cpu_offloading=True \
++model.cpu_offloading_num_layers=${OFFLOAD_NUM_LAYERS} \
++model.cpu_offloading_method="group_async" \
++model.cpu_offloading_region=${OFFLOAD_REGION} \
model.nsys_profile.enabled=True \
model.nsys_profile.gen_shape=True \
model.nsys_profile.start_step=1 \
model.nsys_profile.end_step=1 \
2>&1 | tee nsys_gpt_5b_offload_MBS_${MICRO_BATCH_SIZE}_seq_${SEQ_LENGTH}_amp_O2_${MEGATRON_AMP_O2}_offload_num_layers_${OFFLOAD_NUM_LAYERS}_region_${OFFLOAD_REGION}.log