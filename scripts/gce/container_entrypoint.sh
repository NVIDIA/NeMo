#! /bin/bash

if [ ${USE_GCSFUSE:?} -eq 0 ]; then
  echo "Mounting $GCS_BUCKET to /gcs using gcsfuse"
  gcsfuse ${GCS_BUCKET:?} /gcs
fi

NETWORK_DEVICES=`ifconfig | gawk '{ if (match($0,/(^[^ ].*):/,m)) print m[1] }'`
declare -A NETWORK_DEVICE_TO_TX_COUNTER
declare -A NETWORK_DEVICE_TO_RX_COUNTER

for DEVICE in $NETWORK_DEVICES; do
   TX=`ifconfig $DEVICE | gawk '{ if (match($0,/.*TX.*bytes ([^ ]*) /,m)) print m[1] }'`
   RX=`ifconfig $DEVICE | gawk '{ if (match($0,/.*RX.*bytes ([^ ]*) /,m)) print m[1] }'`
   echo "Found network device $DEVICE with TX counter $TX and RX counter $RX on start"
   NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]=$TX
   NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]=$RX
done

export COMMAND="bash /workspace/nemo/scripts/gce/pretrain_megatron_gpt.sh"

if [ ${PROFILING_ENABLED:?} -eq 0 ]; then
    echo "NVIDIA Nsight profiling is DISABLED."
    eval $COMMAND
elif [ $PROFILING_ENABLED -eq 1 ]; then
    NSIGHT_PATH=${PROFILING_OUTPUT_ROOT_PATH:?}/nsight_${JOB_TIMESTAMP:?}
    mkdir -p $NSIGHT_PATH

    echo "NVIDIA Nsight profiling has been ENABLED and will be emitted to $NSIGHT_PATH."
    nsys profile \
        --sample=none \
        --trace=${PROFILING_TRACE:?} \
        -o $NSIGHT_PATH/node_${NODE_RANK:?} \
        $COMMAND
fi

for DEVICE in $NETWORK_DEVICES; do
   TX=`ifconfig $DEVICE | gawk '{ if (match($0,/.*TX.*bytes ([^ ]*) /,m)) print m[1] }'`
   RX=`ifconfig $DEVICE | gawk '{ if (match($0,/.*RX.*bytes ([^ ]*) /,m)) print m[1] }'`
   echo "Found network device $DEVICE with TX counter $TX and RX counter $RX on end"

   TX_DELTA=`echo "$TX - ${NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]}" | bc`
   RX_DELTA=`echo "$RX - ${NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]}" | bc`
   echo "Transmitted TX $TX_DELTA and RX $RX_DELTA bytes on $DEVICE during workoad"
   echo ""
done
