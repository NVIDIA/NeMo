python scripts/vlm/neva_pretrain.py \
    --devices=2 --data_type=mock --use_toy_model \
    --mbs=2 --gbs=4 --max_steps=4 \
    --tp=2 \
    --log_dir=/tmp/nemo2_neva_results/$RUN_ID
