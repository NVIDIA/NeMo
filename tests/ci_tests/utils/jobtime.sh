#! /bin/bash

set -euo pipefail
shopt -s lastpipe

if [ "$#" -ne 1 ]; then
    exit 1
fi

JOBID="$1"

## Note: using export so that this line doesn't cause the script to immediately exit if the subshell failed when running under set -e
export WALLTIME=$(sacct -j "${JOBID}" --format ElapsedRaw --parsable2 --noheader | head -n 1)

echo ${WALLTIME:-unknown}