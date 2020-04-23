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
# NGC
IMAGE="nvidian/pytorch:19.12-py3"
GPU_MEM=16         # Default is 32.
NUM_GPU=1          # Default is 8.
OPT=O2             # Default is O0.
WS=stan            # Workspace name
WORKSPACE=/ws      # Workspace mount point
RESULT=/result     # Results dir
# LJSpeech
DATASET_SIZE=13036 # Train
NUM_EPOCHS=100     # Total number of epochs

# ---------------------------------------------------- SAVE STATE ----------------------------------------------------
echo "Updating source code..."
# Choose run id.
name_id=$1
num_id=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13)
id=durs-lj_"${name_id}"_"${num_id}"
# Choose tmp dir to save current state of the project.
tmp_dir=/tmp/nemos/"${id}"
echo "Tmp dir: ${tmp_dir}"
# Save current state.
mkdir -p "${tmp_dir}"
rsync -r . "${tmp_dir}" --exclude .git --filter=":- .gitignore"
ngc workspace upload "${WS}" --source "${tmp_dir}" --destination nemos/"${id}"

# -------------------------------------------------- CHOOSE COMMAND --------------------------------------------------
script=examples/tts/fasterspeech_durs.py
config=examples/tts/configs/fasterspeech-durs-lj.yaml
# [1GPU,16G] Biggest bs is 256. One epoch is around 51 iterations. Total number of steps is 5100.
# [1GPU,32G] Biggest bs is 512. One epoch is around 26 iterations. Total number of steps is 2600.
batch_size=$((GPU_MEM * 16))
total_steps=$((((DATASET_SIZE / batch_size) + 1) * NUM_EPOCHS))
read -r -d '' cmd <<EOF
nvidia-smi \
&& apt-get update && apt-get install -y libsndfile1 && pip install -U librosa \
&& cp -R ${WORKSPACE}/nemos/${id} /nemo && cd /nemo && pip install .[all] \
&& pip install -U wandb && wandb login ${WANDB_TOKEN} \
&& python -m torch.distributed.launch --nproc_per_node=${NUM_GPU} ${script} \
--amp_opt_level=${OPT} \
--model_config=${config} \
--batch_size=${batch_size} \
--train_freq=$((total_steps / 500)) \
--eval_freq=$((total_steps / 50)) \
--warmup=$((total_steps / 50)) \
--num_epochs=${NUM_EPOCHS} \
--work_dir=${RESULT} \
--wdb_name=${name_id} \
--wdb_tags=durs,ljspeech,opt \
--train_dataset=/data/ljspeech/train.json \
--train_durs=/data/librimeta/durs/ljspeech_300epochs-qn15x5-eqlen_all-1s/train.npy \
--eval_names=eval \
--eval_datasets=/data/ljspeech/eval.json \
--eval_durs=/data/librimeta/durs/ljspeech_300epochs-qn15x5-eqlen_all-1s/eval.npy
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
  --workspace "${WS}":"${WORKSPACE}" \
  --commandline "${cmd}"
