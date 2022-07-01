#! /bin/bash

set -euo pipefail
shopt -s lastpipe

if [ "$#" -ne 1 ]; then
    exit 1
fi

JOBID="$1"

export STATE=$(bash ngcjobstate.sh "${JOBID}")
case "${STATE}" in
    FAILED)
        echo "JobID ${JOBID} - '${STATE}'..."
        exit 1
        ;;
    *)
        echo "JobID ${JOBID} - '${STATE}'..."
        exit 0
        ;;
esac