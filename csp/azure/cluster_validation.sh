#!/bin/bash

# Path for azure scripts. Can be modified if the repo
# specified in the README is cloned somewhere else
# besides /shared/data
AZURE_NCCL_PATH="/shared/data/azure"

usage() {
cat <<EOF

Validate cluster compute nodes' GPUs and node-to-node communication using 
DCGM Diagnostics and NCCL all_reduce_perf bus bandwidth test

Usage:
  $0 [--OPTION=[VAL] ...]

  OPTION         DESCRIPTION
  --nodelist     List of nodes to run validation on. Can be a comma-separated
                    list or a range such as "hostname-[1-4,6-8]". Same 
                    format as sinfo.
  --nodes        Number of nodes specified in --nodelist
  --partition    Slurm partition of nodes to run validation on. See sinfo for
                    valid partitions.
  --dcgm         Run only DCGM diagnostic
  --nccl         Run only NCCL test

EOF
}

required() {
cat <<EOF

Input error. Required Flags:
  --nodelist
  --nodes
  --partition
EOF
}

error_exit() {
    echo -e "Error: " $1
    exit $2
}

join () {
  local IFS="$1"
  shift
  echo "$*"
}

# Read arguments
while [[ $# -gt 0 ]]; do
  if [[ "$1" =~ ^-.*= ]]; then
    key="${1%%=*}"
    val="${1#*=}"
    val_separate=0
  else
    key="$1"
    val="$2"
    val_separate=1
  fi
  key="$(echo "$key" | tr '[:upper:]' '[:lower:]')"

  case "$key" in
    --nodelist)
      NODES="$val"
      shift $((val_separate+1))
      ;;
    --nodes)
      NUM_NODES="$val"
      shift $((val_separate+1))
      ;;
    --partition)
      PARTITION="$val"
      shift $((val_separate+1))
      ;;
    --dcgm)
      RUN_DCGMI=1
      shift
      ;;
    --nccl)
      RUN_NCCL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      error_exit "Unrecognized option $key." 1
      ;;
  esac
done

# These arguments are required for sbatch commands
if [ -z $NODES ] || [ -z $NUM_NODES ] || [ -z $PARTITION ]; then
    required
    exit 1
fi

# Basic check to ensure valid values for required arguments
srun -p ${PARTITION} -N ${NUM_NODES} -w ${NODES} true > /dev/null 2> /dev/null
if [[ $? != 0 ]]; then
    usage
    error_exit "Invalid values for one of --partition --nodes or --nodelist.\nCheck sinfo." 1
fi

# Enable all checks if none specified
if [ -z $RUN_DCGMI ] && [ -z $RUN_NCCL ]; then
    RUN_DCGMI=1
    RUN_NCCL=1
fi

# Define where test logs should be written to and read from
RESULTS_DIR=../../results/cluster_validation
mkdir -p $RESULTS_DIR

if [[ $RUN_DCGMI == 1 ]]; then
    echo "Starting DCGM Diagnostics..."
    JOBID=$(sbatch  -N $NUM_NODES \
            -p $PARTITION \
            -w $NODES \
            -o $RESULTS_DIR/dcgmi-%j.out \
            -W dcgmi_diag.sh)
    JOBID=${JOBID##* } # Remove everything but the slurm job id from output of sbatch command
    grep -i "fail" $RESULTS_DIR/dcgmi-${JOBID}.out > /dev/null # Check log for failures
    FOUND=$?
    if [[ $FOUND == 0 ]]; then
        # One of the diagnostics failed
        FAILED=$(grep -i -P -o "srun: error: [^:]*" $RESULTS_DIR/dcgmi-${JOBID}.out | cut -d':' -f3)
        echo -e "DCGM failed on the following nodes: \n $FAILED"
        echo "See results/cluster_validation/dcgmi-${JOBID}.out for more details"
        exit 2
    elif [[ $FOUND == 1 ]]; then
        # Something else went wrong
        echo "DCGM diagnostics passing on all nodes!"
    else
        error_exit "DCGM failed to run properly..." 1
    fi
fi

if [[ $RUN_NCCL == 1 ]]; then
    echo "Starting NCCL all_reduce_perf..."
    # Get list of nodes from sinfo to iterate over
    # and create pairwise all_reduce_perf tests
    NODES=$(sinfo --Node -h --partition=${PARTITION} --state=idle --nodes=${NODES})
    NODES_ARR=($NODES)
    ARR_LEN=${#NODES_ARR[@]}
    NCCL_TEST="${AZURE_NCCL_PATH}/benchmarking/NDv4/cc-slurm-ngc/nccl/nccl.sub"

    declare -a slurm_ids # ids for all the jobs launched, should be $NODES / 2
    for (( i = 0; i < $ARR_LEN - 1; i+=8 ))
    do
        j=$((i + 4))
        id=$(sbatch -N 2 --parsable -o $RESULTS_DIR/%x_%j.log -w ${NODES_ARR[$i]},${NODES_ARR[$j]} $NCCL_TEST)
        slurm_ids+=($id)
    done
    all_ids=$(join , "${slurm_ids[@]}")

    # Wait for NCCL jobs to finish
    srun -p ${PARTITION} -N 1 --dependency=$all_ids true > /dev/null 2> /dev/null

    nccl_pass=1
    for i in $slurm_ids; do
        CURR_NODES=$(sed -n 2p $RESULTS_DIR/nccl.sub_${i}.log | cut -d"=" -f2) # Get the nodes in this test
        LARGE_TEST_LINE=$(tail -4 $RESULTS_DIR/nccl.sub_${i}.log | head -1) # Get the results of the 8 GiB size test
        LARGE_TEST_ARR=($LARGE_TEST_LINE)
        CURR_BUSBW=${LARGE_TEST_ARR[6]%.*} # Find the out-of-place bus bandwidth

        # Check CURR_BUSBW is a number
        re='^[0-9]+$'
        if ! [[ $CURR_BUSBW =~ $re ]]; then
            error_exit "NCCL failed to run properly..." 1
        elif [[ $CURR_BUSBW -lt 180 ]]; then
            echo "Insufficient bus bandwidth on nodes $CURR_NODES"
            echo "See results/cluster_validation/nccl.sub_${i}.log for more details"
            nccl_pass=0
        fi
    done
    if [[ $nccl_pass == 0 ]]; then
        # Fail if any nodes had insufficient busbw
        exit 2
    else
        echo "NCCL test passing on all nodes!"
    fi

fi
