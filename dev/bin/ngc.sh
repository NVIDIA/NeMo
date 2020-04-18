#!/usr/bin/env bash

if [[ -z $1 || "$#" -ne 1 ]]; then
  echo "Usage: ./$(basename "$0") [experiment name id]"
  exit 1
fi

if [ ! -f setup.py ]; then
  echo "Script should be ran from project root."
  exit 1
fi

# ------------------------------------------------------ CONSTS ------------------------------------------------------
IMAGE="nvidian/pytorch:19.12-py3"
GPU_MEM=32     # Default is 32.
NUM_GPU=8      # Default is 8.
OPT=O0         # Default is O0.
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
config=examples/tts/configs/fasterspeech.yaml
read -r -d '' cmd <<EOF
nvidia-smi \
&& apt-get update && apt-get install -y libsndfile1 \
&& pip install -U librosa \
&& cp -R ${WORKSPACE}/nemos/${id} /nemo && cd /nemo && pip install .[all] \
&& python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} ${script} \
--amp_opt_level=${OPT} \
--work_dir=${RESULT} \
--model_config=${config} \
--tensorboard_dir=${WORKSPACE}/tb/mels/${id} \
--train_dataset=/manifests/libritts/train-all.json \
--train_durs=/data/librimeta/durs/libritts_original-qn15x5_24k/train-all_full-pad.npy \
--train_speakers=/data/librimeta/speaker-embs/train-all.npy \
--eval_names dev-clean dev-other test-clean test-other \
--eval_datasets \
/manifests/libritts/dev-clean.json \
/manifests/libritts/dev-other.json \
/manifests/libritts/test-clean.json \
/manifests/libritts/test-other.json \
--eval_durs \
/data/librimeta/durs/libritts_original-qn15x5_24k/dev-clean_full-pad.npy \
/data/librimeta/durs/libritts_original-qn15x5_24k/dev-other_full-pad.npy \
/data/librimeta/durs/libritts_original-qn15x5_24k/test-clean_full-pad.npy \
/data/librimeta/durs/libritts_original-qn15x5_24k/test-other_full-pad.npy \
--eval_speakers \
/data/librimeta/speaker-embs/dev-clean.npy \
/data/librimeta/speaker-embs/dev-other.npy \
/data/librimeta/speaker-embs/test-clean.npy \
/data/librimeta/speaker-embs/test-other.npy
EOF

# ------------------------------------------------------- FIRE -------------------------------------------------------
ngc batch run \
  --name "${name_id}" \
  --image "${IMAGE}" \
  --ace nv-us-west-2 \
  --instance dgx1v."${GPU_MEM}"g."${NUM_GPU}".norm \
  --result "${RESULT}" \
  --datasetid 58106:/data/libritts \
  --datasetid 58404:/manifests/libritts \
  --datasetid 59319:/data/librimeta \
  --workspace "${WS}":"${WORKSPACE}" \
  --commandline "${cmd}"
