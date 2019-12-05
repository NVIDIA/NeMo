#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
NEMO_REPO=${NEMO_REPO:-"${SCRIPT_DIR}/../.."}

# Launch TRT JASPER container.

DATA_DIR=${DATA_DIR:-"/datasets"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/checkpoints"}
RESULT_DIR=${RESULT_DIR:-"/results"}
PROGRAM_PATH=${PROGRAM_PATH}
    

MOUNTS=""
if [ ! -z "$DATA_DIR" ]; 
then
    MOUNTS="$MOUNTS -v $DATA_DIR:/datasets "
fi

if [ ! -z "$CHECKPOINT_DIR" ]; 
then
    MOUNTS="$MOUNTS -v $CHECKPOINT_DIR:/checkpoints "
fi

if [ ! -z "$RESULT_DIR" ]; 
then
    MOUNTS="$MOUNTS -v $RESULT_DIR:/results "
fi

echo $MOUNTS
nvidia-docker run -it --rm \
  --runtime=nvidia \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ${MOUNTS} \
  -v ${NEMO_REPO}:/NeMo \
  ${EXTRA_NEMO_ENV} \
  nemo:latest bash $PROGRAM_PATH
