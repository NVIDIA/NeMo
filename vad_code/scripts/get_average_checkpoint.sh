#!/bin/bash
curr_dir=${pwd}

proj_name="Frame_VAD"
exp_dir="drc_Multilang_sgdlr1e-2_wd1e-3_augx_b512_gacc1_ep50_ns_wce_td_n4"

proj_dir=/gpfs/fs1/projects/ent_aiapps/users/heh/results/${proj_name}
source_dir=${proj_dir}/${exp_dir}/${exp_dir}/checkpoints/
target_dir=./nemo_experiments/${proj_name}/${exp_dir}/

mkdir -p ${target_dir}

rsync -Wav heh@draco1:${source_dir} ${target_dir}

python checkpoint_averaging.py ${target_dir}
