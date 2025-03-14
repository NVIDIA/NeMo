coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/neural_diarizer/sortformer_diar_train.py \
    trainer.devices="[0]" \
    batch_size=3 \
    model.train_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_train/eesd_train_tiny.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_valid/eesd_valid_tiny.json \
    exp_manager.exp_dir=/tmp/speaker_diarization_results \
    +trainer.fast_dev_run=True
