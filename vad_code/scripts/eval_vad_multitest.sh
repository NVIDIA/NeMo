
# proj_name="Stream_VAD"
# exp_dir="drc_Multilang_sgdlr1e-3_wd1e-4_augx_b128_gacc1_ep50_w8"
# ckpt_dir="./nemo_experiments/${proj_name}/${exp_dir}"

# exp_dir="marblenet_3x2x64_mandarin_40ms_all"
exp_dir="marblenet_3x2x64_multilang_40ms_all_noise"

ckpt_dir="./nemo_experiments/${exp_dir}/checkpoints"

model_path="${ckpt_dir}/${exp_dir}-averaged.nemo"
data_dir=./manifests_test

python infer_vad_multi.py \
    --config-path="./configs" --config-name="vad_inference" \
    vad.model_path=$model_path \
    frame_out_dir="${ckpt_dir}/frame_vad_multi_output" \
    dataset="[${data_dir}/ava_eval_10ms.json,${data_dir}/ami_eval_10ms.json,${data_dir}/ch120_CH109_10ms.json]"

    # dataset="[${data_dir}/french_test_20ms.json,${data_dir}/german_test_20ms.json,${data_dir}/mandarin_test_20ms.json,${data_dir}/russian_test_20ms.json,${data_dir}/spanish_test_20ms.json]"
