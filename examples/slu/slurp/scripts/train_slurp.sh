DATA_DIR="./slurp_data"
CUDA_VISIBLE_DEVICES=1 python run_slurp_train.py \
    --config-path="./configs" --config-name=conformer_transformer_bpe \
    model.train_ds.manifest_filepath="[${DATA_DIR}/train_slu.json,${DATA_DIR}/train_synthetic_slu.json]" \
    model.validation_ds.manifest_filepath="${DATA_DIR}/devel_slu.json" \
    model.test_ds.manifest_filepath="${DATA_DIR}/test_slu.json" \
    model.tokenizer.dir="${DATA_DIR}/tokenizers_slu/tokenizer_spe_unigram_v58_pad_bos_eos" \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=false
