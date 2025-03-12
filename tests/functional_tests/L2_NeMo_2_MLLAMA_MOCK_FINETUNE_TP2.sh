TRANSFORMERS_OFFLINE=1 \
    torchrun --nproc_per_node=2 /opt/NeMo/scripts/vlm/mllama_finetune.py \
    --devices=2 --data_type=mock --use_toy_model \
    --mbs=2 --gbs=4 --max_steps=4 \
    --tp=2 \
    --log_dir=/tmp/nemo2_mllama_results/$RUN_ID
