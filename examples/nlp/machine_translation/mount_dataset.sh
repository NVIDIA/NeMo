#!/bin/bash

INSTANCE=dgx1v.32g.8.norm
PROJECT=nmt-de-en-ngc
EXPNAME=STUDENT_NMT_DE_EN_DISTILL_NGC
DATAID=81837

ngc batch run --name ${EXPNAME} \
    --image "nvcr.io/nvidia/pytorch:21.05-py3" \
    --ace nv-us-west-2 \
    --instance ${INSTANCE} \
    --result /results/ \
    --org nvidian \
    --team ac-aiapps \
    --datasetid ${DATAID}:/data/ \
    --commandline "cp -R /data/* /raid/"