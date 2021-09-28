#!/bin/bash

WORKSPACE="/workspace"
MODEL_PATH=${PWD}/fastertransformer_backend/all_models
IMAGE=nvcr.io#nvidian/swdl-joc/ft_triton:latest

MODEL_FILENAME=530B
BATCH_SIZE=1
INPUT_LEN=512
OUTPUT_LEN=16
SIZE_PER_HEAD=160
HEAD_NUM=128
VOCAB_SIZE=51200
NUM_DECODER_LAYERS=105
NUM_RUNS=1
SERVER_TIMEOUT=420
PIPELINE_PARA_SIZE=3
MAX_SEQ_LEN=$(( $INPUT_LEN + $OUTPUT_LEN ))
INTER_SIZE=$(($HEAD_NUM * $SIZE_PER_HEAD * 4))
MODEL_NAME=$MODEL_FILENAME

stage=0

function usage
{
    echo "Usage: $0 [--stage <stage>]"
    echo "Options: "
    echo "    --stage <stage>              # controls partial re-runs"
}

while [ "$1" != "" ]; do
    case $1 in
	-s | --stage) shift
		      stage=$1
		      ;;
	-h | --help)  shift
	              usage
		      exit 1
		      ;;
	* )           usage
		      exit 1
    esac
    shift
done


# copy code to workspace
if [ $stage -le 0 ]; then
   cp -r ../fastertransformer_backend .
fi

# build
if [ $stage -le 1 ]; then
    CMD="mkdir $WORKSPACE/fastertransformer_backend/build/;cd $WORKSPACE/fastertransformer_backend/build/ && \
        cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .. && \
        make -j12;cp $WORKSPACE/fastertransformer_backend/build/libtriton_fastertransformer.so $WORKSPACE/fastertransformer_backend/build/lib/libtransformer-shared.so /opt/tritonserver/backends/fastertransformer"
    srun -N1 -A joc -p interactive --container-mounts ${PWD}:/workspace --container-image $IMAGE bash -c "$CMD"

fi

# start server
if [ $stage -le 2 ]; then

(cd $MODEL_PATH/fastertransformer && \
    echo '
name: "fastertransformer"
backend: "fastertransformer"
default_model_filename: "'"${MODEL_FILENAME}"'"
max_batch_size: '"${BATCH_SIZE}"'
input [
  {
    name: "INPUT_ID"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: "REQUEST_INPUT_LEN"
    data_type: TYPE_UINT32
    dims: [ 1 ]
  },
  {
    name: "REQUEST_OUTPUT_LEN"
    data_type: TYPE_UINT32
    dims: [ 1 ]
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind : KIND_CPU
  }
]
parameters {
  key: "top_k"
  value: {
    string_value: "0"
  }
}
parameters {
  key: "top_p"
  value: {
    string_value: "0.9"
  }
}
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "8"
  }
}
parameters {
  key: "pipeline_para_size"
  value: {
    string_value: "'"${PIPELINE_PARA_SIZE}"'"
  }
}
parameters {
  key: "max_input_len"
  value: {
    string_value: "'"${INPUT_LEN}"'"
  }
}
parameters {
  key: "max_seq_len"
  value: {
    string_value: "'"${MAX_SEQ_LEN}"'"
  }
}
parameters {
  key: "is_half"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "head_num"
  value: {
    string_value: "'"${HEAD_NUM}"'"
  }
}
parameters {
  key: "size_per_head"
  value: {
    string_value: "'"${SIZE_PER_HEAD}"'"
  }
}
parameters {
  key: "inter_size"
  value: {
    string_value: "'"${INTER_SIZE}"'"
  }
}
parameters {
  key: "vocab_size"
  value: {
    string_value: "'"${VOCAB_SIZE}"'"
  }
}
parameters {
  key: "start_id"
  value: {
    string_value: "50256"
  }
}
parameters {
  key: "end_id"
  value: {
    string_value: "50256"
  }
}
parameters {
  key: "decoder_layers"
  value: {
    string_value: "'"${NUM_DECODER_LAYERS}"'"
  }
}
parameters {
  key: "model_name"
  value: {
    string_value: "'"${MODEL_NAME}"'"
  }
}
parameters {
  key: "max_batch_size"
  value: {
    string_value: "'"${BATCH_SIZE}"'"
  }
}
parameters {
  key: "beam_width"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "temperature"
  value: {
    string_value: "1.0"
  }
}
parameters {
  key: "repetition_penalty"
  value: {
    string_value: "1.0"
  }
}
parameters {
  key: "len_penalty"
  value: {
    string_value: "1.0"
  }
}
parameters {
  key: "beam_search_diversity_rate"
  value: {
    string_value: "0.0"
  }
}
' > config.pbtxt)

RES=$(sbatch start_server.slurm)
JOB_ID=${RES##* }
echo "Server job id ${JOB_ID}"


CMD="python $WORKSPACE/fastertransformer_backend/tools/identity_test.py -b 1 -s 512 -o 32 -r -v"

while :
do
    status=$(squeue -j $JOB_ID | awk '{ print $5 }' | tail -n1)
    echo $status
    if [ $status == "R" ]; then
	echo "Waiting server to start..."
	sleep 8m #TODO: check server is finish loading by curl
	tail -n100 output.${JOB_ID}
	srun -N1 --jobid ${JOB_ID} --overlap --container-name multi-node-ft-triton --container-mounts ${PWD}:/workspace --pty bash -c "$CMD"
	#scancel $JOB_ID
	break
    else
	echo "Waiting nodes to be allocated.."
	sleep 10s
    fi
done

fi
