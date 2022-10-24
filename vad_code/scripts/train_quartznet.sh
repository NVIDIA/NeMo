DATA_DIR="/media/data/datasets/mandarin/aishell2"
NAME="quartznet_15x5_mandarin_40ms"
CUDA_VISIBLE_DEVICES=0,1 python speech_to_multi_label.py \
    --config-path="./configs" --config-name="quartznet_15x5" \
    model.train_ds.manifest_filepath="${DATA_DIR}/manifests_abs/svad_mandarin_40ms_train.json" \
    model.validation_ds.manifest_filepath="[${DATA_DIR}/manifests_abs/svad_mandarin_40ms_dev.json,${DATA_DIR}/manifests_abs/svad_mandarin_40ms_dev.json]" \
    model.test_ds.manifest_filepath="${DATA_DIR}/manifests_abs/svad_mandarin_40ms_dev.json" \
    model.train_ds.batch_size=64 \
    model.validation_ds.batch_size=32 \
    model.test_ds.batch_size=32 \
    trainer.devices=2 \
    trainer.max_epochs=50 \
    exp_manager.name=${NAME} \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project="Stream_VAD"
