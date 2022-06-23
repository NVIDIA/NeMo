#! /bin/bash

shopt -s lastpipe

if [ "$#" -ne 1 ]; then
    exit 1
fi

JOBID="$1"

sacct -j "${JOBID}" --format State --parsable2 --noheader |& head -n 1

exit ${PIPESTATUS[0]}
