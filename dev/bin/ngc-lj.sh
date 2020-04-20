#!/usr/bin/env bash

if [[ -z $1 || "$#" -ne 1 ]]; then
  echo "Usage: ./$(basename "$0") [experiment name id]"
  exit 1
fi

if [ ! -f setup.py ]; then
  echo "Script should be ran from project root."
  exit 1
fi

if [[ -z $WANDB_TOKEN ]]; then
  echo "Please, provide WanDB token."
  exit 1
fi

# ------------------------------------------------------ CONSTS ------------------------------------------------------
IMAGE="nvidian/pytorch:19.12-py3"
GPU_MEM=32     # Default is 32. Could be also 16 for <20M models with bs=64.
NUM_GPU=1      # Default is 8.
OPT=O2         # Default is O0.
WS=stan        # Workspace name.
WORKSPACE=/ws  # Workspace mount point.
RESULT=/result # Results dir.

# ---------------------------------------------------- SAVE STATE ----------------------------------------------------
echo "Updating source code..."
# Choose run id.
name_id=$1
num_id=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13)
id="${name_id}"_"${num_id}"
# Choose tmp dir to save current state of the project.
tmp_dir=/tmp/nemos/"${id}"
echo "Tmp dir: ${tmp_dir}"
# Save current state.
mkdir -p "${tmp_dir}"
rsync -r . "${tmp_dir}" --exclude .git --filter=":- .gitignore"
ngc workspace upload "${WS}" --source "${tmp_dir}" --destination nemos/"${id}"

# -------------------------------------------------- CHOOSE COMMAND --------------------------------------------------
script=examples/tts/fasterspeech.py
config=examples/tts/configs/fasterspeech-lj.yaml
# One epoch is around 200 iterations. Total number of steps is 20400.
read -r -d '' cmd <<EOF
nvidia-smi \
&& apt-get update && apt-get install -y libsndfile1 && pip install -U librosa \
&& cp -R ${WORKSPACE}/nemos/${id} /nemo && cd /nemo && pip install .[all] \
&& pip install -U wandb && wandb login ${WANDB_TOKEN} \
&& python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} ${script} \
--amp_opt_level=${OPT} \
--model_config=${config} \
--train_freq=$((NUM_GPU * 20)) \
--eval_freq=$((NUM_GPU * 200)) \
--warmup=$((NUM_GPU * 300)) \
--num_epochs=$((NUM_GPU * 100)) \
--work_dir=${RESULT} \
--wdb_name=${name_id} \
--wdb_tags=ljspeech,mel,opt \
--train_dataset=/data/ljspeech/train.json \
--train_durs=/data/librimeta/durs/ljspeech_300epochs-qn15x5-eqlen_all-1s/train.npy \
--eval_names=eval \
--eval_datasets=/data/ljspeech/eval.json \
--eval_durs=/data/librimeta/durs/ljspeech_300epochs-qn15x5-eqlen_all-1s/eval.npy \
--waveglow_checkpoint=/data/checkpoints/waveglow.pth
EOF

# ------------------------------------------------------- FIRE -------------------------------------------------------
ngc batch run \
  --name "${name_id}" \
  --image "${IMAGE}" \
  --ace nv-us-west-2 \
  --instance dgx1v."${GPU_MEM}"g."${NUM_GPU}".norm \
  --result "${RESULT}" \
  --datasetid 59558:/data/ljspeech \
  --datasetid 59319:/data/librimeta \
  --datasetid 59662:/data/checkpoints \
  --workspace "${WS}":"${WORKSPACE}" \
  --commandline "${cmd}"
