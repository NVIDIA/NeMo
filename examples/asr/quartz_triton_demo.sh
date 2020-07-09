#!/bin/bash
#set -Eueo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

WAV=${1:-"${DIR}/../../tests/data/asr/train/an4/wav/an127-mcen-b.wav"}
JASPER_REPO="${DIR}/triton/repository/jasper-asr-streaming-vad"

if [ ! -f "${JASPER_REPO}/jasper-trt-encoder-streaming/1/model.plan" ] ; then
    pkill trtserver
    echo "Exporting models ..."
    ${DIR}/export_quartz_to_trt.sh
fi

sleep 1
server_pid=$(pgrep trtserver)

if [ "${server_pid}" == "" ] ; then 
    echo "Starting TRITON server, log is in server.log ..."
    trtserver --log-verbose=100 --model-control-mode=none --model-repository=${JASPER_REPO} > server.log 2>&1 &
    sleep 100
else
    echo "TRITON server is running with PID ${server_pid}"
fi 

python3 ${DIR}/test_asr.py --chunk_duration 0.5 --wav $WAV

# mv server.log server.log.last
