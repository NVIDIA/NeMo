#! /bin/bash

set -euo pipefail
shopt -s lastpipe

if [ "$#" -ne 1 ]; then
    exit 1
fi

JOBID="$1"

while true; do
    export STATE=$(bash jobstate.sh "${JOBID}")
    case "${STATE}" in
        PENDING|RUNNING|REQUEUED)
            sleep 15s
            ;;
        *) 
            sleep 30s
            echo "Exiting with SLURM job status '${STATE}'"
            exit 0
            ;;
    esac
done
