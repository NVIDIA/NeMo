#!/usr/bin/env bash

if [[ -z $1 || "$#" -ne 1 ]]; then
  echo "Usage: ./$(basename "$0") [experiment name id]"
  exit 1
fi

if [ ! -f setup.py ]; then
  echo "Script should be run from project root."
  exit 1
fi

# ------------------------------------------------------ CONSTS ------------------------------------------------------
IMAGE="nvidian/pytorch:19.12-py3"
GPU_MEM=32    # Default is 32.
NUM_GPU=8     # Default is 8.
WS=stan       # Workspace name.
WORKSPACE=/ws # Workspace mount point.
RESULT=/result

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
script=examples/asr/quartznet.py
config=examples/asr/configs/qn-libritts.yaml
read -r -d '' cmd <<EOF
nvidia-smi \
&& apt-get update && apt-get install -y libsndfile1 \
&& cp -R ${WORKSPACE}/nemos/${id} /nemo && cd /nemo && pip install .[all] \
&& pip install -U wandb && wandb login ${WANDB_TOKEN} \
&& python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} ${script} \
--num_epochs=400 \
--eval_batch_size=32 \
--model_config=${config} \
--checkpoint_dir=${RESULT} \
--checkpoint_save_freq=10000 \
--load_dir=/data/ckpts/qn-multien \
--lr=0.001 \
--warmup_steps=20000 \
--weight_decay=0.001 \
--beta2=0.25 \
--train_dataset=/manifests/libritts/train-all.json \
--eval_datasets \
/manifests/libritts/dev-clean.json \
/manifests/libritts/dev-other.json \
/manifests/libritts/test-clean.json \
/manifests/libritts/test-other.json \
/data/ljspeech/split3/eval.json \
/data/ljspeech/split3/test.json
EOF

# ------------------------------------------------------- FIRE -------------------------------------------------------
ngc batch run \
  --name "${name_id}" \
  --image "${IMAGE}" \
  --ace nv-us-west-2 \
  --instance dgx1v."${GPU_MEM}"g."${NUM_GPU}".norm \
  --result "${RESULT}" \
  --datasetid 58106:/data/libritts \
  --datasetid 59943:/data/ljspeech \
  --datasetid 58404:/manifests/libritts \
  --datasetid 48230:/data/ckpts/qn-multien \
  --workspace "${WS}":"${WORKSPACE}" \
  --commandline "${cmd}"
