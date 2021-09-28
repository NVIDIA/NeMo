#!/bin/bash
#SBATCH -o output.%j
#SBATCH -vvv
#SBATCH -A joc
#SBATCH -p backfill
#SBATCH -N3

IMAGE=nvcr.io#nvidian/swdl-joc/ft_triton:latest
WORKSPACE="/workspace"
CMD="cp $WORKSPACE/fastertransformer_backend/build/libtriton_fastertransformer.so $WORKSPACE/fastertransformer_backend/build/lib/libtransformer-shared.so /opt/tritonserver/backends/fastertransformer;/opt/tritonserver/bin/tritonserver -\
-model-repository=$WORKSPACE/fastertransformer_backend/all_models"


srun -N3 -n3 --mpi=pmix --container-mounts ${PWD}:/workspace --container-name multi-node-ft-triton --container-image $IMAGE bash -c "$CMD"

