# DATA_DIR="/media/data/datasets/mandarin/aishell2"
# DATA_DIR="/media/data/projects/NeMo-vad/project/manifests_local"
DATA_DIR="./manifests_local"
NAME="marblenet_3x2x64_multilang_40ms_all_debug2_pm"
BATCH_SIZE=256
NUM_WORKERS=4
PIN_MEMORY=true
CUDA_VISIBLE_DEVICES=0,1 python speech_to_multi_label.py \
    --config-path="./configs" --config-name="marblenet_3x2x64" \
    model.train_ds.manifest_filepath="[${DATA_DIR}/ami_train_40ms_local.json,${DATA_DIR}/fisher_2004_40ms_local.json,${DATA_DIR}/fisher_2005_40ms_local.json,${DATA_DIR}/icsi_all_40ms_local.json,${DATA_DIR}/french_train_40ms_cleaned_local.json,${DATA_DIR}/german_train_40ms_local.json,${DATA_DIR}/mandarin_train_40ms_local.json,${DATA_DIR}/russian_train_40ms_local.json,${DATA_DIR}/spanish_train_40ms_local.json]" \
    model.validation_ds.manifest_filepath="[${DATA_DIR}/ami_dev_40ms_local.json,${DATA_DIR}/ch120_moved_40ms_local.json,${DATA_DIR}/french_dev_40ms_local.json,${DATA_DIR}/german_dev_40ms_local.json,${DATA_DIR}/mandarin_dev_40ms_local.json,${DATA_DIR}/russian_dev_40ms_local.json,${DATA_DIR}/spanish_dev_40ms_local.json]" \
    model.test_ds.manifest_filepath="[${DATA_DIR}/ami_dev_40ms_local.json,${DATA_DIR}/ch120_moved_40ms_local.json,${DATA_DIR}/french_dev_40ms_local.json,${DATA_DIR}/german_dev_40ms_local.json,${DATA_DIR}/mandarin_dev_40ms_local.json,${DATA_DIR}/russian_dev_40ms_local.json,${DATA_DIR}/spanish_dev_40ms_local.json]" \
    model.train_ds.batch_size=$BATCH_SIZE \
    model.validation_ds.batch_size=$BATCH_SIZE \
    model.test_ds.batch_size=$BATCH_SIZE \
    model.train_ds.num_workers=$NUM_WORKERS \
    model.validation_ds.num_workers=$NUM_WORKERS \
    model.test_ds.num_workers=$NUM_WORKERS \
    model.train_ds.pin_memory=$PIN_MEMORY \
    model.validation_ds.pin_memory=$PIN_MEMORY \
    model.test_ds.pin_memory=$PIN_MEMORY \
    trainer.max_epochs=50 \
    exp_manager.name=${NAME} \
    exp_manager.create_wandb_logger=false \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project="Frame_VAD"


# with pin_memory=true, training failed at the first iteration
# with pin_memory=false, training works for some time
