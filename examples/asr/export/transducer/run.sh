python infer_transducer_onnx.py \
     --pretrained_model="stt_en_conformer_transducer_large" \
    --export \
    --dataset_manifest="/home/dgalvez/scratch/data/test_clean.json" \
    --max_symbold_per_step=5 \
    --batch_size=16 \
    --log
