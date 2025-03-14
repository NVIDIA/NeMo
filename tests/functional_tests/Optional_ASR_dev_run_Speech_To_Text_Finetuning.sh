coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/speech_to_text_finetune.py \
    --config-path="conf/asr_finetune" --config-name="speech_to_text_finetune" \
    model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    init_from_nemo_model=/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo \
    model.tokenizer.update_tokenizer=False \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_finetuning_results
