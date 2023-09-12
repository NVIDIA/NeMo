NEMO=/home/scratch.guyueh_sw/2023su/cpu_offload/NeMo
export PYTHONPATH=${NEMO}:$PYTHONPATH

nsys \
profile -s none -t nvtx,cuda -o ./gh_nsys_sanity_check --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python ${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
model.data.data_impl=mock model.data.data_prefix=[] \
model.nsys_profile.enabled=True \
model.nsys_profile.start_step=1 \
model.nsys_profile.end_step=1 \
model.nsys_profile.gen_shape=True \
2>&1 | tee gh_nsys_sanity_check.log