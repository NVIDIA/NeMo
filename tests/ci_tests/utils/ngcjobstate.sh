#! /bin/bash

shopt -s lastpipe

if [ "$#" -ne 1 ]; then
    exit 1
fi

JOBID="$1"

ngc batch info $JOBID | jq -r ".jobStatus.status"

exit ${PIPESTATUS[0]}
