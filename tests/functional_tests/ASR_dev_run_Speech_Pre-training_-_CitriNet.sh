coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/speech_pretraining/speech_pre_training.py \
    --config-path="../conf/ssl/citrinet/" --config-name="citrinet_ssl_ci" \
    model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_pre_training_results
