#! /bin/bash

shopt -s lastpipe

if [ "$#" -ne 1 ]; then
    exit 1
fi

JOBID="$1"

sacct -j "${JOBID}" -n --format=exitcode | sort -r -u | head -1 | cut -f 1 -d":" | sed 's/ //g'

exit ${PIPESTATUS[0]}