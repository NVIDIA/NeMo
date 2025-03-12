python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
--config-path="../conf/squeezeformer" --config-name="squeezeformer_ctc_bpe" \
model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
model.tokenizer.type="wpe" \
model.encoder.d_model=144 \
model.train_ds.batch_size=4 \
model.validation_ds.batch_size=4 \
trainer.devices=1 \
trainer.accelerator="gpu" \
+trainer.fast_dev_run=True \
exp_manager.exp_dir=/tmp/speech_to_text_wpe_squeezeformer_results
