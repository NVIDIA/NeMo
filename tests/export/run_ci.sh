echo "unset all SLURM_, PMI_, PMIX_ Variables"
set -x
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done
set +x

python3 /opt/NeMo/tests/export/test_nemo_export_ci.py \
         --model_name=$MODEL_NAME \
         --model_type=$MODEL_TYPE \
         --n_gpus=$NUM_GPUS \
         --location=$LOCATION \
         --trt_llm_model_dir=$TRT_LLM_MODEL_DIR \
         --checkpoint=$CHECKPOINT \
         --max_output_token=$MAX_OUTPUT_TOKEN \
         --max_batch_size=$MAX_BATCH_SIZE \
         --p_tuning_checkpoint=$P_TUNING_CHECKPOINT
