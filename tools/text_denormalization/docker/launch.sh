#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
: ${CLASSIFY_DIR:="$SCRIPT_DIR/classify"}
: ${VERBALIZE_DIR:="$SCRIPT_DIR/verbalize"}
: ${CMD:=${1:-/bin/bash}}

MOUNTS=""
MOUNTS+=" -v $CLASSIFY_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/classify"
MOUNTS+=" -v $VERBALIZE_DIR:/workspace/sparrowhawk/documentation/grammars/en_toy/verbalize"

echo $MOUNTS
docker run -it --rm \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $MOUNTS \
  -w /workspace/sparrowhawk/documentation/grammars \
  sparrowhawk:latest $CMD