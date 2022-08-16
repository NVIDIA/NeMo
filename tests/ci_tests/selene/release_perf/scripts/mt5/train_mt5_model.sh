set -o xtrace

# Default values
MAX_STEPS=100
DATA_DIR=/lustre/fsw/swdl/swdl-langspeech/datasets/data/BigNLP/mC4
PREPROCESSED_DIR=
PRECISION=${PRECISION:-bf16}
CREATE_CHECKPOINT_CALLBACK_FLAG=False
TOKENIZER_MODEL=/lustre/fsw/swdl/swdl-langspeech/datasets/data/BigNLP/mC4/mt5_tokenizer.model

case $RUN_MODEL_SIZE in

  170m)
    NUM_NODES=${NUM_NODES:-4}
    TP_SIZE=${TP_SIZE:-1}
    PP_SIZE=${PP_SIZE:-1}
    ;;

  390m)
    NUM_NODES=${NUM_NODES:-8}
    TP_SIZE=${TP_SIZE:-1}
    PP_SIZE=${PP_SIZE:-1}
    ;;

  3b)
    NUM_NODES=${NUM_NODES:-20}
    TP_SIZE=${TP_SIZE:-2}
    PP_SIZE=${PP_SIZE:-1}
    ;;

  11b)
    NUM_NODES=${NUM_NODES:-20}
    TP_SIZE=${TP_SIZE:-4}
    PP_SIZE=${PP_SIZE:-1}
    ;;

  23b)
    NUM_NODES=${NUM_NODES:-40}
    TP_SIZE=${TP_SIZE:-4}
    PP_SIZE=${PP_SIZE:-2}
    ;;

  *)
    echo -n "unknown"
    ;;
esac

LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
LIMIT_VAL_BATCHES=`expr $MAX_STEPS / 20`

if [[ $AMP_STYLE = O1 ]]; then
  AMP_O2_FLAG=False
else
  AMP_STYLE=O2
  AMP_O2_FLAG=True
fi

export RUN_NAME=${RUN_MODEL}_${RUN_MODEL_SIZE}_tp${TP_SIZE}_pp${PP_SIZE}_${NUM_NODES}nodes_${PRECISION}_precision_${AMP_STYLE}_${MAX_STEPS}steps
export RESULTS_DIR=${BASE_RESULTS_DIR}/${RUN_NAME}

HYDRA_FULL_ERROR=1 python3 main.py \
    training=${RUN_MODEL}/${RUN_MODEL_SIZE} \
    stages=["training"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${DATA_DIR} \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    training.run.name=${RUN_NAME} \
    training.run.time_limit=${TIME_LIMIT} \
    training.run.preprocessed_dir=${DATA_DIR} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.trainer.max_steps=${MAX_STEPS} \
    training.trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
    training.trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    training.trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    training.trainer.precision=${PRECISION} \
    training.model.tensor_model_parallel_size=${TP_SIZE} \
    training.model.pipeline_model_parallel_size=${PP_SIZE} \
    training.model.megatron_amp_O2=${AMP_O2_FLAG} \
    training.model.tokenizer.model=${TOKENIZER_MODEL} \
    training.exp_manager.create_checkpoint_callback=${CREATE_CHECKPOINT_CALLBACK_FLAG}
