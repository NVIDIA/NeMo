#! /bin/bash
set -e
set -u
set -o pipefail

: "${MASTER_ADDR:?Must set MASTER_ADDR}"
: "${NODE_RANK:?Must set NODE_RANK}"
: "${GCS_BUCKET:?Must set GCS_BUCKET}"
: "${JOB_TIMESTAMP:?Must set JOB_TIMESTAMP}"
: "${EXPERIMENT_ROOT_DIR:?Must set EXPERIMENT_ROOT_DIR}"
: "${CONFIG_FILE:?Must set CONFIG_FILE}"


gcsfuse --max-conns-per-host 65535 "$GCS_BUCKET" /gcs

export MASTER_PORT=6002
GPUS_PER_NODE=8
NNODES=$(yq -r ".trainer.num_nodes" "${EXPERIMENT_ROOT_DIR}/config/${CONFIG_FILE}")
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

cat ${EXPERIMENT_ROOT_DIR}/config/${CONFIG_FILE}

PROFILING_DIR=$EXPERIMENT_ROOT_DIR/nsys_profiles
mkdir -p $PROFILING_DIR

LOG_DIR=$EXPERIMENT_ROOT_DIR/training_logs
mkdir -p $LOG_DIR

EXPERIMENTS_DIR=$EXPERIMENT_ROOT_DIR/nemo_experiments
mkdir -p $EXPERIMENTS_DIR

DEBUG_DIR=$EXPERIMENT_ROOT_DIR/debug
mkdir -p $DEBUG_DIR

export NCCL_TOPO_DUMP_FILE=$DEBUG_DIR/nccl_topo_${JOB_TIMESTAMP}.xml

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=12 



wait_all_success_or_exit() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pids=("$@")
  while [[ ${#pids[@]} -ne 0 ]]; do
    all_success="true"
    for pid in "${pids[@]}"; do
      code=$(non_blocking_wait "$pid")
      if [[ $code -ne 127 ]]; then
        if [[ $code -ne 0 ]]; then
          echo "PID $pid failed with exit code $code"
          exit "$code"
        fi
      else
        all_success="false"
      fi
    done
    if [[ $all_success == "true" ]]; then
      echo "All pids succeeded"
      break
    fi
    sleep 5
  done
}

non_blocking_wait() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pid=$1
  local code=127 # special code to indicate not-finished
  if [[ ! -d "/proc/$pid" ]]; then
    wait "$pid"
    code=$?
  fi
  echo $code
}

PIDS=()

for ((LOCAL_RANK=0; LOCAL_RANK <= $((GPUS_PER_NODE - 1)); LOCAL_RANK++)); do
   RANK=$(($GPUS_PER_NODE*$NODE_RANK + $LOCAL_RANK))
   
    RANK=$RANK LOCAL_RANK=$LOCAL_RANK \
    nsys profile --sample=none --trace=cuda,nvtx -o $PROFILING_DIR/node_${NODE_RANK:?}_local_rank_${LOCAL_RANK} \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
     python /workspace/nemo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
      --config-path="$EXPERIMENT_ROOT_DIR/config" \
      --config-name="$CONFIG_FILE" \
      > >(tee "$LOG_DIR/pretrain_gpt_rank$RANK.log") 2>&1 &
   PID=$!
   PIDS+=($PID)

   echo "Launched pretrain_gpt.py for rank $RANK with PID $PID"
done

wait_all_success_or_exit "${PIDS[@]}"

