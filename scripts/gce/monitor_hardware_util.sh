#!/bin/bash

# Metric Kind one of GAUGE or CUMALATIVE
# Metric Type one of DOUBLE or INT64
# see https://cloud.google.com/monitoring/api/v3/kinds-and-types
declare -A METRIC_ID_TO_KIND
declare -A METRIC_ID_TO_TYPE

declare -A NETWORK_DEVICE_TO_TX_COUNTER
declare -A NETWORK_DEVICE_TO_RX_COUNTER

main_loop() {
   perform_initialization "$@"

   SAMPLE_INDEX=0
   while [[ ! -f "/tmp/workload_terminated" ]]; do
      get_auth_token
      record_sample_start_time
      emit_device_utilization
      stall_until_sample_end_time

      ((SAMPLE_INDEX++))
   done
}

perform_initialization() {
  # Note: The rate-limiting step is Cloud Monitoring (min period of 5 sec)
  # https://cloud.google.com/monitoring/quotas
  SAMPLE_PERIOD=${DEVICE_COLLECTION_INTERVAL:-8}
  echo "Device monitoring sample period set to $SAMPLE_PERIOD seconds"

  record_global_start_time
  detect_environment
  create_metrics

  print_banner
  initialize_nic_counters
  enable_nvidia_persistence_mode
}

record_global_start_time() {
   GLOBAL_START_TIME=$(date +%s)
   GLOBAL_ISO8601_START_TIME=$(date -Iseconds)
   echo "Global start time at $(date) ($GLOBAL_START_TIME)"
}

detect_environment() {
   local now
   now="$(date '+%Y-%m-%d-%H-%M-%S')"

   NODE_RANK="${NODE_RANK:-0}"
   JOB_TIMESTAMP="${JOB_TIMESTAMP:-$now}"

   get_vm_metadata
   get_auth_token

   detect_cpus
   detect_gpus
   detect_nics
}

create_metrics() {
   create_metric "cpu/load" "GAUGE" "DOUBLE" "1" "cpu_number"
   create_metric "gpu/load" "GAUGE" "DOUBLE" "1" "gpu_number"
   create_metric "gpu/memory" "GAUGE" "INT64" "By" "gpu_number"
   create_metric "nic/traffic" "CUMULATIVE" "INT64" "By" "interface" "direction"
}

print_banner() {
  echo "Warning: monitor_hardware_util.sh uses simple expressions to parse to output of cli tools!"
  echo "Warning: monitor_hardware_util.sh may be brittle to minor changes. If in doubt, confirm query."
}

initialize_nic_counters() {
   declare -A identifiers
   for DEVICE in $NETWORK_DEVICES; do
      TX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*TX.*bytes ([^ ]*) /,m)) print m[1] }')
      RX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*RX.*bytes ([^ ]*) /,m)) print m[1] }')
      echo "Found network device $DEVICE with counters TX $TX and RX $RX bytes"

      NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]=$TX
      NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]=$RX      
   done
   NIC_CAPTURE_TIME=$(date +%s)
}
  
enable_nvidia_persistence_mode() {
   nvidia-smi -pm 1 
}

get_vm_metadata() {
   METADATA_SERVER='metadata.google.internal'
   METADATA_HEADER=(-H "Metadata-Flavor: Google")

   INSTANCE_ID_URI="http://$METADATA_SERVER/computeMetadata/v1/instance/id"
   INSTANCE_ID=$(curl -s "$INSTANCE_ID_URI" "${METADATA_HEADER[@]}")
   echo "Instance ID is $INSTANCE_ID"

   INSTANCE_ZONE_URI="http://$METADATA_SERVER/computeMetadata/v1/instance/zone"
   INSTANCE_ZONE=$(curl -s "$INSTANCE_ZONE_URI" "${METADATA_HEADER[@]}" | xargs basename)
   echo "Instance zone is $INSTANCE_ZONE"

   PROJECT_ID_URI="http://$METADATA_SERVER/computeMetadata/v1/project/project-id"
   PROJECT_ID=$(curl -s "$PROJECT_ID_URI" "${METADATA_HEADER[@]}")
   echo "Project is $PROJECT_ID"
}

get_auth_token() {
   METADATA_SERVER='metadata.google.internal'
   METADATA_HEADER=(-H "Metadata-Flavor: Google")

   ACCESS_TOKEN_URI="http://$METADATA_SERVER/computeMetadata/v1/instance/service-accounts/default/token"
   ACCESS_TOKEN=$(curl -s $ACCESS_TOKEN_URI "${METADATA_HEADER[@]}" | jq -r .access_token)
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

create_metric() {
   PROJECT_SPECIFIER="projects/$PROJECT_ID"
   MONITORING_SERVER="monitoring.googleapis.com"
   METRIC_DESCRIPTOR_URI="https://$MONITORING_SERVER/v3/$PROJECT_SPECIFIER/metricDescriptors"

   local metric_id="$1"
   local metric_kind="$2"
   local metric_type="$3"
   local metric_unit="$4"
   shift 4
   local addl_labels=($@)

   local base_labels=("node_rank" "job_timestamp")
   local metric_labels=("${base_labels[@]}" "${addl_labels[@]}")

   labels_json=""
   for index in "${!metric_labels[@]}"; do
     if (( index > 0 )); then
        labels_json="$labels_json"','
     fi
     labels_json="$labels_json"'
     {
       "key": "'"${metric_labels["$index"]}"'",
       "valueType": "STRING",
       "description": "'"${metric_labels["$index"]}"'"
     }'
   done

   local create_metric_body='{
     "name": "",
     "description": "NeMO '"$metric_id"'",
     "displayName": "NeMO '"$metric_id"'",
     "type": "workload.googleapis.com/nemo/'$metric_id'",
     "metricKind": "'"$metric_kind"'",
     "valueType": "'"$metric_type"'",
     "unit": "'"$metric_unit"'",
     "labels": [
       '"$labels_json"'
     ],
   }'

   curl --silent --output /dev/null --show-error \
      -X POST "$METRIC_DESCRIPTOR_URI" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d "$create_metric_body"

   METRIC_ID_TO_KIND["$metric_id"]=$metric_kind
   METRIC_ID_TO_TYPE["$metric_id"]=$metric_type
}

record_sample_start_time() {
   SAMPLE_START_TIME=$(date +%s)
}

emit_device_utilization() {
   emit_cpu_utilization
   emit_gpu_utilization
   emit_nic_utilization
}

stall_until_sample_end_time() {
   local SAMPLE_END_TIME=$(( (SAMPLE_INDEX+1) * SAMPLE_PERIOD))

   update_elapsed_time
   local DELAY=$((SAMPLE_END_TIME-GLOBAL_ELAPSED_TIME))
   if (( DELAY > 0 )); then
      sleep $DELAY
   fi
}

emit_cpu_utilization() {
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
   declare -A identifiers

   local ROW_INDEX=0
   local CPU_INDEX=""
   while read -ra ROW; do
       local COL_INDEX=0
       for VALUE in "${ROW[@]}"; do
           case $ROW_INDEX in
              "2")
                 # Warning: Take care in indexing of column to metric
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
   CPU_LOAD=$(echo "scale=2; $CPU_COUNT * $AVG_LOAD / 100.0" | bc | awk '{printf "%f", $0}')

   identifiers["cpu_number"]="all"
   emit_metric "cpu/load" "$CPU_LOAD" identifiers
}

emit_gpu_utilization() {
   declare -a SMI_OPTIONS

   local SMI_COMMAND="nvidia-smi"
   local SMI_OPTIONS=(dmon -c 1 -s um)
   if [[ "$SAMPLE_INDEX" == "0" ]]; then
      echo "Detecting GPU utilization using \"$SMI_COMMAND ${SMI_OPTIONS[*]}\""
   fi

   declare -A COL_INDEX_TO_METRIC_ID
   declare -A COL_INDEX_TO_METRIC_UNIT

   update_elapsed_time
   declare -A identifiers

   local ROW_INDEX=0
   local GPU_INDEX=0
   while read -ra ROW; do
       local COL_INDEX=0
       for VALUE in "${ROW[@]}"; do
           case $ROW_INDEX in
              "0")
                 # Warning: Take care in indexing of column to metric
                 COL_INDEX_TO_METRIC_ID[$((COL_INDEX-1))]=$VALUE
                 ;;
              "1")
                 COL_INDEX_TO_METRIC_UNIT[$((COL_INDEX-1))]=$VALUE
                 ;;
              *)
                 METRIC_ID=${COL_INDEX_TO_METRIC_ID[$COL_INDEX]}
                 case $METRIC_ID in
                    "gpu") 
                       identifiers["gpu_number"]="$VALUE"
                       ;;
                    "sm")  
                       GPU_LOAD=$(echo "scale=2; $VALUE / 100"| bc | awk '{printf "%f", $0}')
                       emit_metric "gpu/load" "$GPU_LOAD" identifiers 
                       ;;
                    "fb")  
                       emit_metric "gpu/memory" "$(echo "$VALUE * 1024 * 1024" | bc)" identifiers 
                       ;;
                 esac
                 ;;
           esac

           (( COL_INDEX++ ))
       done
       (( ROW_INDEX++ ))
   done < <("$SMI_COMMAND" "${SMI_OPTIONS[@]}")
}

emit_nic_utilization() {
   update_elapsed_time
   declare -A identifiers

   for DEVICE in $NETWORK_DEVICES; do
      TX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*TX.*bytes ([^ ]*) /,m)) print m[1] }')
      RX=$(/usr/sbin/ifconfig "$DEVICE" | gawk '{ if (match($0,/.*RX.*bytes ([^ ]*) /,m)) print m[1] }')

      TX_DELTA=$(echo "$TX - ${NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]}" | bc)
      RX_DELTA=$(echo "$RX - ${NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]}" | bc)

      identifiers["interface"]="$DEVICE"

      identifiers["direction"]="tx"
      emit_metric "nic/traffic" "$TX_DELTA" identifiers

      identifiers["direction"]="rx"
      emit_metric "nic/traffic" "$RX_DELTA" identifiers

      NETWORK_DEVICE_TO_TX_COUNTER[$DEVICE]=$TX
      NETWORK_DEVICE_TO_RX_COUNTER[$DEVICE]=$RX      
   done
   NIC_CAPTURE_TIME=$(date +%s)
}

update_elapsed_time() {
   (( GLOBAL_ELAPSED_TIME=$(date +%s) - $GLOBAL_START_TIME))
   (( SAMPLE_ELAPSED_TIME=$(date +%s) - $SAMPLE_START_TIME))
}

emit_metric() {
   PROJECT_SPECIFIER="projects/$PROJECT_ID"
   MONITORING_SERVER="monitoring.googleapis.com"
   TIME_SERIES_URI="https://$MONITORING_SERVER/v3/$PROJECT_SPECIFIER/timeSeries"

   local metric_id="$1"
   local metric_value="$2"
   local -n metric_labels=$3

   metric_labels["node_rank"]="$NODE_RANK"
   metric_labels["job_timestamp"]="$JOB_TIMESTAMP"

   build_metric_labels_json metric_labels
   build_metric_resource_json
   build_metric_interval_json "$metric_id"
   build_metric_value_json "$metric_id" "$metric_value"

   local emit_metric_json='{
     "timeSeries": [
      {
       "metric": {
        "type": "workload.googleapis.com/nemo/'$metric_id'",
        "labels": '"$LABELS_JSON"'
       },
       "resource": '"$RESOURCE_JSON"',
       "points": [
        {
         "interval": '"$INTERVAL_JSON"',
         "value": '"$VALUE_JSON"'
        }
       ]
      }
     ]
   }'

   curl --silent --output /dev/null --show-error \
      -X POST "$TIME_SERIES_URI" \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      -d "$emit_metric_json"
}

build_metric_labels_json() {
   local -n label_ids=$1

   LABELS_JSON='{ '
   for label in "${!label_ids[@]}"; do
     LABELS_JSON=$LABELS_JSON'"'"$label"'": "'"${label_ids["$label"]}"'",'
   done
   LABELS_JSON=${LABELS_JSON::-1}'}'
}

build_metric_resource_json() {
   RESOURCE_JSON='{
     "type": "gce_instance",
     "labels": {
        "project_id": "'"$PROJECT_ID"'",
        "instance_id": "'"$INSTANCE_ID"'",
        "zone": "'"$INSTANCE_ZONE"'"
      }
   }'
}

build_metric_interval_json() {
   local metric_id=$1

   INTERVAL_JSON='{'
   if [[ "${METRIC_ID_TO_KIND["$metric_id"]}" == "CUMULATIVE" ]]; then
     INTERVAL_JSON="$INTERVAL_JSON"'"startTime": "'"$GLOBAL_ISO8601_START_TIME"'",'
   fi
   INTERVAL_JSON="$INTERVAL_JSON"' "endTime": "'"$(date -Iseconds)"'"'
   INTERVAL_JSON="$INTERVAL_JSON"'}'
}

build_metric_value_json() {
   local metric_id=$1
   local metric_value=$2

   VALUE_JSON='{'
   if [[ "${METRIC_ID_TO_TYPE["$metric_id"]}" == "INT64" ]]; then
      VALUE_JSON=$VALUE_JSON'"int64Value": '"$metric_value"
   else
      VALUE_JSON=$VALUE_JSON'"doubleValue": '"$metric_value"
   fi
   VALUE_JSON=$VALUE_JSON'}'
}

main_loop "$@"; exit
