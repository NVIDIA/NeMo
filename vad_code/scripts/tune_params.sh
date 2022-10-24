
# proj_name="Stream_VAD"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b128_gacc1_ep50_w8"
# ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

exp_dir="marblenet_3x2x64_mandarin_40ms_all"
split="ami_dev_10ms"

ckpt_dir="./nemo_experiments/${exp_dir}/checkpoints"
output_dir="${ckpt_dir}/frame_vad_dev_output/vad_output_${split}"
pred_dir="${output_dir}/frames_predictions"
gt_dir="${output_dir}/frames_groundtruth"

python run_grid_search.py \
    pred_dir=$pred_dir \
    gt_dir=$gt_dir
