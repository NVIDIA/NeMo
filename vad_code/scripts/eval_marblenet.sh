
proj_name="Stream_VAD"
exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b128_gacc1_ep50_w8"

ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"
model_path="${ckpt_dir}/${exp_dir}-averaged.nemo"

python infer_vad.py \
    --config-path="./configs" --config-name="vad_inference" \
    vad.model_path=$model_path \
    frame_out_dir="$ckpt_dir/frame_vad_output" \
    num_workers=12 \
    dataset="./manifests_test/french_test_20ms.json"
