#! /bin/bash

set -euo pipefail
shopt -s lastpipe

if [ "$#" -ne 1 ]; then
    exit 1
fi

JOBID="$1"

sacct -j "${JOBID}"  -X -n --format=nodelist%400 | sed 's/ //g'
