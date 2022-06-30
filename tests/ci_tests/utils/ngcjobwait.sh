#! /bin/bash

set -euo pipefail
shopt -s lastpipe

if [ "$#" -ne 2 ]; then
    exit 1
fi

JOBID="$1"
WAIT_RUNNING="$2"
WAIT_RUNNING=$(($WAIT_RUNNING + 0))
while true; do
    export STATE=$(bash ngcjobstate.sh "${JOBID}")
    case "${STATE}" in
        CREATED|QUEUED|STARTING)
            sleep 15s
            ;;
        RUNNING)
            if [ $WAIT_RUNNING -ne 0 ]; then
              sleep 15s
            else
              sleep 5s
              echo "JobID ${JOBID} - '${STATE}'..."
              exit 0
            fi
            ;;
        *)
            sleep 5s
            echo "JobID ${JOBID} - '${STATE}'..."
            exit 0
            ;;
    esac
done