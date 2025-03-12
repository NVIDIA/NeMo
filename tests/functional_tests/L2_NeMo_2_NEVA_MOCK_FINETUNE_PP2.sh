python scripts/vlm/neva_finetune.py \
--devices=2 --data_type=mock --use_toy_model \
--mbs=2 --gbs=4 --max_steps=4 \
--pp=2 \
--log_dir=/tmp/nemo2_neva_results/${{ github.run_id }}
