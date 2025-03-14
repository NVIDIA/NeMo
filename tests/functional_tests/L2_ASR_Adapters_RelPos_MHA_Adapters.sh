coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_adapters/train_asr_adapter.py \
    model.pretrained_model="stt_en_conformer_ctc_small" \
    model.adapter.adapter_name="encoder:an4" \
    model.adapter.adapter_type="tiny_attn" \
    model.adapter.tiny_attn.n_feat=176 \
    model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    trainer.max_steps=5 \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_to_text_adapters_mha_results
