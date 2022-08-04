DATA_DIR="./slurp_data"
EXP_NAME="ssl_en_conformer_large_transformer_CosineAnneal_adamwlr3e-4x2e-4_wd0_dec3_d2048h8"
CKPT_DIR="./nemo_experiments/${EXP_NAME}/checkpoints"

python ../../../scripts/checkpoint_averaging/checkpoint_averaging.py ${CKPT_DIR}

NEMO_MODEL="${CKPT_DIR}/${EXP_NAME}-averaged.nemo"
CUDA_VISIBLE_DEVICES=0 python run_slurp_eval.py \
    dataset_manifest="${DATA_DIR}/test_slu.json" \
    model_path=${NEMO_MODEL} \
    batch_size=32 \
    num_workers=8 \
    searcher.type="beam" \
    searcher.beam_size=32 \
    searcher.temperature=1.25 \
    only_score_manifest=false