#!/bin/bash

# This script is used to submit multiple consecutive SLURM jobs with different random seeds.
# Usage:
# ./auto_launcher_with_seed.sh -n4 <file.sub>
# Where <file.sub> is a SLURM job submission script that takes one argument.
# It will call each file with one argument that's corresponding to the global random seed base,
# e.g. "file.sub 1545" -> "file.sub 2547657" -> "file.sub 3547657" -> ...
# The global random seed base is used to synchronize the data sampling RNGs across all jobs,
# ensuring each data-parallel-rank gets a different slice of data, 
# and each tensor-parallel-rank gets the same slice.

# Grab command line options
# n: Number of times to submit the job
N_CALLS=1
while getopts "n:" opt; do
  case $opt in
    n) N_CALLS=$OPTARG;;
  esac
done

# Shift the command line arguments to get the SUBFILE
shift $((OPTIND-1))
SUBFILE=$1
if [[ -z $SUBFILE ]]; then
  echo "Usage: $(basename "$0") [flags] [sub file] [arguments for sub file]"
  exit 1
fi

# Remove the SUBFILE from the argument list
shift

echo "Calling [$SUBFILE] $N_CALLS times"

# Repeat calls
PREV_JOBID=""
for (( i = 1; i <= $N_CALLS; i++ ))
do
  RSEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')
  if [ -z $PREV_JOBID ]; then
    echo "Submitting job ${i}"
    OUTPUT=$(sbatch $SUBFILE "$RSEED")
  else
    echo "Submitting job ${i} w/ dependency on jobid ${PREV_JOBID}"
    OUTPUT=$(sbatch --dependency=afterany:${PREV_JOBID} $SUBFILE "$RSEED")
  fi
  PREV_JOBID="$(cut -d' ' -f4 <<< $OUTPUT)"
done

squeue --start -u $(whoami) -l

