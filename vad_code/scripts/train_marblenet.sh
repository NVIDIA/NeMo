# DATA_DIR="/media/data/datasets/mandarin/aishell2"
DATA_DIR="/media/data/projects/NeMo-vad/project/tools/manifests_long"
NAME="marblenet_3x2x64_mandarin_40ms_long_debug"
BATCH_SIZE=1024
CUDA_VISIBLE_DEVICES=0,1 python speech_to_multi_label.py \
    --config-path="./configs" --config-name="marblenet_3x2x64" \
    model.train_ds.manifest_filepath="[${DATA_DIR}/ami_train_40ms.json,${DATA_DIR}/fisher_2004_40ms.json,${DATA_DIR}/fisher_2005_40ms.json,${DATA_DIR}/icsi_all_40ms.json]" \
    model.validation_ds.manifest_filepath="[${DATA_DIR}/ami_dev_40ms.json,${DATA_DIR}/ch120_moved_40ms.json]" \
    model.test_ds.manifest_filepath="${DATA_DIR}/ami_eval_40ms.json" \
    model.train_ds.batch_size=$BATCH_SIZE \
    model.validation_ds.batch_size=$BATCH_SIZE \
    model.test_ds.batch_size=$BATCH_SIZE \
    trainer.max_epochs=50 \
    exp_manager.name=${NAME} \
    exp_manager.create_wandb_logger=false \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project="Stream_VAD"
