#!/usr/bin/env bash

if [[ -z $1 || "$#" -ne 1 ]]; then
  echo "Usage: ./$(basename "$0") [experiment name id]"
  exit 1
fi

# ------------------------------------------------------ CONSTS ------------------------------------------------------
IMAGE="nvidian/pytorch:19.12-py3"
GPU_MEM=32 # Default is 32.
NUM_GPU=8  # Default is 8.
WS=stan
RESULT=/result

# ---------------------------------------------------- SAVE STATE ----------------------------------------------------
echo "Updating source code..."
# Choose run id.
name_id=$1
num_id=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13)
id="${name_id}"-"${num_id}"
# Choose tmp dir to save current state of the project.
tmp_dir=/tmp/nemos/"${id}"
echo "Tmp dir: ${tmp_dir}"
# Save current state.
mkdir -p "${tmp_dir}"
rsync -r . "${tmp_dir}" --exclude .git --filter=":- .gitignore"
ngc workspace upload "${WS}" --source "${tmp_dir}" --destination nemos/"${id}"

# -------------------------------------------------- CHOOSE COMMAND --------------------------------------------------
script=examples/tts/fasterspeech_durations.py
config=examples/tts/configs/fasterspeech_durations.yaml
read -r -d '' cmd <<EOF
nvidia-smi \
&& apt-get update \
&& apt-get install -y libsndfile1 \
&& cp -R /ws/nemos/${id} /nemo && cd /nemo && pip install .[all] \
&& echo "Starting training..." \
&& python -m torch.distributed.launch --nproc_per_node=8 ${script} \
--work_dir=${RESULT} \
--model_config=${config} \
--tensorboard_dir=/ws/tb/${id} \
--train_dataset=/manifests/librispeech/librivox-train-all.json \
--durs_dir=/data/durs
EOF

# ------------------------------------------------------- FIRE -------------------------------------------------------
ngc batch run \
  --name "${name_id}" \
  --image "${IMAGE}" \
  --ace nv-us-west-2 \
  --instance dgx1v."${GPU_MEM}"g."${NUM_GPU}".norm \
  --result "${RESULT}" \
  --datasetid 9367:/data/librispeech \
  --datasetid 32028:/manifests/librispeech \
  --datasetid 56321:/data/durs \
  --workspace "${WS}":/ws \
  --commandline "${cmd}"
