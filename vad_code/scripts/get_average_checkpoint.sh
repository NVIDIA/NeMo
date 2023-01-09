#!/bin/bash
curr_dir=${pwd}

proj_name="Frame_VAD"
# exp_dir="drc_Multilang_sgdlr1e-2_wd1e-3_augx_b512_gacc1_ep50_ns_wce_td_n4"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b512_gacc1_ep100_ns_w8_sfish"
# exp_dir="drc_conformer_small_Multilang_adamwlr5.0_wd1e-3_aug5x0.05_b32_gacc1_ep50_ns_wce"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep100_sf_ns_wce_n4"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-3_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_ep50"
# exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_ns_wce_n8_fnorm_ep50"
exp_dir="drc_marblenet_3x2x64_Multilang_sgdlr1e-2minlr1e-4_wd1e-3_aug10x0.05_b512_gacc1_ep50_sf_trns_wce_n8"

proj_dir=/gpfs/fs1/projects/ent_aiapps/users/heh/results/${proj_name}
source_dir=${proj_dir}/${exp_dir}/${exp_dir}/checkpoints/
target_dir=./nemo_experiments/${proj_name}/${exp_dir}/

mkdir -p ${target_dir}

rsync -Wav heh@draco1:${source_dir} ${target_dir}

python checkpoint_averaging.py ${target_dir}
