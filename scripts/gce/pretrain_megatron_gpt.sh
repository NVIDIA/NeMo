#! /bin/bash

# GKE needs to inject the following environment variables for PyTorch distributed launcher
# MASTER_ADDR, MASTER_PORT, NNODES, NODE_RANK, and WORLD_SIZE

PARALLEL_CONFIG="tensor_${PARALLELISM_TENSOR_SPLITS:?}_pipeline_${PARALLELISM_PIPELINE_STAGES:?}"
PARALLEL_FOLDER=${TRAINING_LOG_ROOT_PATH:?}/${PARALLEL_CONFIG}

# CHECKPOINT_PATH=${TRAINING_SAVE_ROOT_PATH:?}/checkpoints_${JOB_TIMESTAMP:?}/
LOG_PATH=$PARALLEL_FOLDER/logs_$JOB_TIMESTAMP

mkdir -p /sharedfs/training-data &&\
  mkdir -p $PARALLEL_FOLDER &&\
  mkdir -p $NEMO_EXPERIMENT_ROOT_PATH &&\
  mkdir -p $LOG_PATH

if [[ "${NODE_RANK:?}" -ne 0 ]]; then
   SLEEP_DELAY=20
   echo "Not Node 0; sleeping for $SLEEP_DELAY seconds to allow Node 0 to start up"
   sleep $SLEEP_DELAY
fi

cd /workspace/nemo

for ((LOCAL_RANK=0; LOCAL_RANK <= $((GPUS_PER_NODE - 1)); LOCAL_RANK++)); do
   RANK=$(($GPUS_PER_NODE*$NODE_RANK + $LOCAL_RANK))

   OMP_NUM_THREADS=12 RANK=$RANK LOCAL_RANK=$LOCAL_RANK \
     python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
      --config-path=/workspace/nemo/examples/nlp/language_modeling/conf \
      --config-name=megatron_gpt_config \
      trainer.devices=$GPUS_PER_NODE \
      trainer.num_nodes=$NNODES \
      trainer.max_epochs=null \
      trainer.max_steps=${TRAINING_ITERATIONS:?} \
      trainer.val_check_interval=${TRAINING_EVALUATE_INTERVAL:?} \
      trainer.log_every_n_steps=${TRAINING_LOG_INTERVAL:?} \
      trainer.limit_val_batches=${EVALUATE_ITERATIONS:?} \
      trainer.limit_test_batches=50 \
      trainer.accumulate_grad_batches=1 \
      trainer.precision=${PRECISION:?} \
      model.micro_batch_size=${TRAINING_MICRO_BATCH_SIZE:?} \
      model.global_batch_size=${TRAINING_GLOBAL_BATCH_SIZE:?} \
      model.tensor_model_parallel_size=$PARALLELISM_TENSOR_SPLITS \
      model.pipeline_model_parallel_size=$PARALLELISM_PIPELINE_STAGES \
      model.max_position_embeddings=${TRANSFORMER_MAX_POSITION_EMBEDDINGS:?} \
      model.encoder_seq_length=${TRANSFORMER_SEQUENCE_LENGTH:?} \
      model.hidden_size=${TRANSFORMER_HIDDEN_SIZE:?} \
      model.ffn_hidden_size=${FFN_HIDDEN_SIZE:?} \
      model.num_layers=${TRANSFORMER_NUM_LAYERS:?} \
      model.num_attention_heads=${TRANSFORMER_NUM_ATTENTION_HEADS:?} \
      model.init_method_std=0.021 \
      model.hidden_dropout=0.1 \
      model.layernorm_epsilon=1e-5 \
      model.tokenizer.vocab_file=gpt2-vocab.json \
      model.tokenizer.merge_file=gpt2-merges.txt \
      model.data.data_prefix=${TRAINING_DATA_FILE_PREFIX:?} \
      model.data.num_workers=${NUM_DATA_WORKERS:?} \
      model.data.seq_length=${TRANSFORMER_SEQUENCE_LENGTH:?} \
      model.data.splits_string=\'949,50,1\' \
      model.optim.name=fused_adam \
      model.optim.lr=6e-4 \
      model.optim.betas=[0.9,0.95] \
      model.optim.weight_decay=0.1 \
      model.optim.sched.name=CosineAnnealing \
      model.optim.sched.warmup_steps=750 \
      model.optim.sched.constant_steps=80000 \
      model.optim.sched.min_lr=6e-5 \
      model.transformer_engine=${TRANSFORMER_ENGINE_ENABLED:=False} \
      model.nsys_profile.enabled=${NSYS_PROFILE_ENABLED:=False} \
      model.nsys_profile.start_step=${NSYS_PROFILE_START_STEP:=10} \
      model.nsys_profile.end_step=${NSYS_PROFILE_END_STEP:=10} \
      model.nsys_profile.ranks=${NSYS_PROFILE_RANKS:="[0]"} \
      exp_manager.exp_dir=${NEMO_EXPERIMENT_ROOT_PATH:?} \
      +exp_manager.version=${JOB_TIMESTAMP:?} \
      exp_manager.resume_if_exists=True \
      exp_manager.resume_ignore_no_checkpoint=True \
      exp_manager.create_checkpoint_callback=True \
      exp_manager.checkpoint_callback_params.monitor=val_loss \
      exp_manager.checkpoint_callback_params.save_top_k=3 \
      exp_manager.checkpoint_callback_params.mode=min \
      exp_manager.checkpoint_callback_params.always_save_nemo=False 2>&1 | tee $LOG_PATH/pretrain_gpt_rank$RANK.log &
   
   echo "Launched pretrain_megatron_gpt.py for rank $RANK with PID $!"
done

wait
