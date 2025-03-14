coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/speech_classification/speech_to_label.py \
    model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
    model.validation_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
    model.test_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    model.preprocessor._target_=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
    ~model.preprocessor.window_size \
    ~model.preprocessor.window_stride \
    ~model.preprocessor.window \
    ~model.preprocessor.n_mels \
    ~model.preprocessor.n_mfcc \
    ~model.preprocessor.n_fft \
    exp_manager.exp_dir=/tmp/speech_to_label_results
