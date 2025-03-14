coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder.py \
    model.diarizer.speaker_embeddings.model_path=titanet_large \
    model.train_ds.batch_size=5 \
    model.validation_ds.batch_size=5 \
    model.train_ds.emb_dir=examples/speaker_tasks/diarization/speaker_diarization_results \
    model.validation_ds.emb_dir=examples/speaker_tasks/diarization/speaker_diarization_results \
    model.train_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_train/msdd_data.50step.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_valid/msdd_data.50step.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speaker_diarization_results
