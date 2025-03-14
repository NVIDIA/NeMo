coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/recognition/speaker_reco.py \
    model.train_ds.batch_size=10 \
    model.validation_ds.batch_size=2 \
    model.train_ds.manifest_filepath=/home/TestData/an4_speaker/train.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_speaker/dev.json \
    model.decoder.num_classes=2 \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speaker_recognition_results
