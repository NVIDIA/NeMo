#!/bin/bash
# Use json_payload.message!~"^\[device-monitor\].*" to omit these log messages
# Use json_payload.message=~"^\[device-monitor\].*" to filter device trace only

exec >  >(trap "" INT TERM; sed 's/^/[device-monitor]: /')
exec 2> >(trap "" INT TERM; sed 's/^/[device-monitor]: /' >&2)

declare -A NETWORK_DEVICE_TO_TX_COUNTER
declare -A NETWORK_DEVICE_TO_RX_COUNTER

main_loop() {
   SAMPLE_PERIOD=10
   echo "Target sample period set to $SAMPLE_PERIOD seconds"

   print_banner
   detect_devices
   initialize_counters
   record_global_start_time

   SAMPLE_INDEX=0
   while [[ ! -f "/tmp/workload_terminated" ]]; do
      record_sample_start_time
      print_device_utilization
      stall_until_sample_end_time
      ((SAMPLE_INDEX++))
   done
}

print_banner() {
  echo "Warning: monitor_hardware_util.sh uses simple expressions to parse to output of cli tools!"
  echo "Warning: monitor_hardware_util.sh may be brittle to minor changes. If in doubt, confirm query."
  echo "Use json_payload.message=~\"^\\[device-monitor\\].*\" to filter device trace"
}

detect_devices() {
   detect_cpus
   detect_gpus
   detect_nics
}

initialize_counters() {
   initialize_nic_counters
}

print_device_utilization() {
   print_cpu_utilization
   print_gpu_utilization
   print_nic_utilization
}

stall_until_sample_end_time() {
   local SAMPLE_END_TIME=$(( (SAMPLE_INDEX+1) * SAMPLE_PERIOD))

   update_elapsed_time
   local DELAY=$((SAMPLE_END_TIME-GLOBAL_ELAPSED_TIME))
   if (( DELAY > 0 )); then
      sleep $DELAY
   fi
}

detect_cpus() {
   local QUERY="nproc"
   CPU_COUNT=$($QUERY)

   echo "Detected $CPU_COUNT cores using \"$QUERY\""
}

detect_gpus() {
   local QUERY="nvidia-smi --list-gpus | wc -l"
   GPU_COUNT=$(eval "$QUERY")

   echo "Detected $GPU_COUNT GPUs using \"$QUERY\""
}

detect_nics() {
   local IFC_QUERY="/usr/sbin/ifconfig"
   local AWK_QUERY="gawk '{ if (match(\$0,/(^[^ ].*):/,m)) print m[1] }'"
   local AWK_FILTER="gawk '{ if (index(\$0, \"en\") == 1 || index(\$0, \"eth\") == 1) {print \$0} }'"
   local QUERY="$IFC_QUERY | $AWK_QUERY | $AWK_FILTER"
   NETWORK_DEVICES=$(eval "$QUERY")

   echo "Detecting networking devices using \"$QUERY\""
   for DEVICE in $NETWORK_DEVICES; do
      echo "Found network device $DEVICE"
   done
}

initialize_nic_counters() {
   for DEVICE in $NETWORK_DEVICES; do
      TX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*TX.*bytes ([^ ]*) /,m)) print m[1] }')
      RX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*RX.*bytes ([^ ]*) /,m)) print m[1] }')
      echo "Found network device $DEVICE has counters TX $TX bytes and RX $RX bytes on start"

      NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]=$TX
      NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]=$RX
   done
   NIC_CAPTURE_TIME=$(date +%s)
}

record_global_start_time() {
   GLOBAL_START_TIME=$(date +%s)
   echo "Global start time at $(date) ($GLOBAL_START_TIME)"
}

record_sample_start_time() {
   SAMPLE_START_TIME=$(date +%s)
}

update_elapsed_time() {
   (( GLOBAL_ELAPSED_TIME=$(date +%s) - $GLOBAL_START_TIME))
   (( SAMPLE_ELAPSED_TIME=$(date +%s) - $SAMPLE_START_TIME))
}

print_cpu_utilization() {
   local CPU_SAMPLE_PERIOD=1
   local CPU_SAMPLE_COUNT=1
   declare -a MPS_OPTIONS

   local MPS_ENVIRON="S_TIME_FORMAT=ISO"
   local MPS_COMMAND="mpstat"
   local MPS_OPTIONS=(-u "$CPU_SAMPLE_PERIOD" "$CPU_SAMPLE_COUNT")

   if [[ "$SAMPLE_INDEX" == "0" ]]; then
      echo "Detecting CPU utilization using \"$MPS_COMMAND ${MPS_OPTIONS[*]}\""
   fi

   declare -A COL_INDEX_TO_METRIC_ID
   declare -A CPU_INDEX_TO_CPU_LOAD

   update_elapsed_time
   local ROW_INDEX=0
   local CPU_INDEX=""
   while read -ra ROW; do
       local COL_INDEX=0
       for VALUE in "${ROW[@]}"; do
           case $ROW_INDEX in
              "2")
                 COL_INDEX_TO_METRIC_ID[$((COL_INDEX))]="$VALUE"
                 ;;
              *)
                 METRIC_ID="${COL_INDEX_TO_METRIC_ID[$COL_INDEX]}"
                 case "${METRIC_ID}" in
                    "CPU")     CPU_INDEX="$VALUE";;
                    "%idle")   CPU_INDEX_TO_IDLE["$CPU_INDEX"]="$VALUE";;
                    "%iowait") CPU_INDEX_TO_WAIT["$CPU_INDEX"]="$VALUE";;
                 esac
                 ;;
           esac
           (( COL_INDEX++ ))
       done
       (( ROW_INDEX++ ))
   done < <(export "$MPS_ENVIRON"; "$MPS_COMMAND" "${MPS_OPTIONS[@]}")

   ALL_IDLE=$(echo "scale=2; ${CPU_INDEX_TO_IDLE["all"]} + ${CPU_INDEX_TO_WAIT["all"]}" | bc)
   AVG_LOAD=$(echo "scale=2; 100.0 - $ALL_IDLE" | bc)
   CPU_LOAD=$(echo "scale=2; $CPU_COUNT * $AVG_LOAD / 100.0" | bc)

   printf "# CPU %10s %10s %15s\n" "Time(s)" "CPU-index" "CPU-load"
   printf "$ CPU %10d %10s %15.2f\n" "$GLOBAL_ELAPSED_TIME" "all" "$CPU_LOAD"
}

print_gpu_utilization() {
   declare -a SMI_OPTIONS

   local SMI_COMMAND="nvidia-smi"
   local SMI_OPTIONS=(dmon -c 1 -s um)
   if [[ "$SAMPLE_INDEX" == "0" ]]; then
      echo "Detecting GPU utilization using \"$SMI_COMMAND ${SMI_OPTIONS[*]}\""
   fi

   declare -A COL_INDEX_TO_METRIC_ID
   declare -A COL_INDEX_TO_METRIC_UNIT

   update_elapsed_time
   printf "# GPU %10s %10s %15s %15s" "Time(s)" "GPU-index" "GPU-load" "GPU-mem(By)"
   local ROW_INDEX=0
   while read -ra ROW; do
       local COL_INDEX=0
       for VALUE in "${ROW[@]}"; do
           case $ROW_INDEX in
              "0")
                 COL_INDEX_TO_METRIC_ID[$((COL_INDEX-1))]=$VALUE
                 ;;
              "1")
                 COL_INDEX_TO_METRIC_UNIT[$((COL_INDEX-1))]=$VALUE
                 ;;
              *)
                 METRIC_ID=${COL_INDEX_TO_METRIC_ID[$COL_INDEX]}
                 case $METRIC_ID in
                    "gpu") printf "\n$ GPU %10d %10d " "$GLOBAL_ELAPSED_TIME" "$VALUE";;
                    "sm")  printf "%15.2f " "$(echo "scale=2; $VALUE / 100"| bc)";;
                    "fb")  printf "%15d " "$(echo "$VALUE * 1024 * 1024" | bc)";;
                 esac
                 ;;
           esac

           (( COL_INDEX++ ))
       done
       (( ROW_INDEX++ ))
   done < <("$SMI_COMMAND" "${SMI_OPTIONS[@]}")
   printf "\n"
}

print_nic_utilization() {
   update_elapsed_time
   printf "# NIC %10s %10s %15s %15s\n" "Time(s)" "NIC-name" "NIC-tx(By/s)" "NIC-rx(By/s)"

   for DEVICE in $NETWORK_DEVICES; do
      TX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*TX.*bytes ([^ ]*) /,m)) print m[1] }')
      RX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*RX.*bytes ([^ ]*) /,m)) print m[1] }')

      TX_DELTA=$(echo "$TX - ${NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]}" | bc)
      RX_DELTA=$(echo "$RX - ${NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]}" | bc)
      ((TIME_DELTA=$(date +%s) - $NIC_CAPTURE_TIME))

      TX_RATE=$(echo "scale=2; $TX_DELTA / $TIME_DELTA" | bc)
      RX_RATE=$(echo "scale=2; $RX_DELTA / $TIME_DELTA" | bc)
      printf "$ NIC %10d %10s %15.1f %15.1f\n" "$GLOBAL_ELAPSED_TIME" "$DEVICE" "$TX_RATE" "$RX_RATE"

      NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]=$TX
      NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]=$RX
   done
   NIC_CAPTURE_TIME=$(date +%s)
}

main_loop "$@"; exit
