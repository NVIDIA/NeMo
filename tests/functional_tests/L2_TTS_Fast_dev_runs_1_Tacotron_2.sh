coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/tacotron2.py \
    train_dataset=/home/TestData/an4_dataset/an4_train.json \
    validation_datasets=/home/TestData/an4_dataset/an4_val.json \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
    trainer.strategy=auto \
    model.decoder.decoder_rnn_dim=256 \
    model.decoder.attention_rnn_dim=1024 \
    model.decoder.prenet_dim=128 \
    model.postnet.postnet_n_convolutions=3 \
    model.train_ds.dataloader_params.batch_size=4 \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.batch_size=4 \
    model.validation_ds.dataloader_params.num_workers=0 \
    ~model.text_normalizer \
    ~model.text_normalizer_call_kwargs \
    ~trainer.check_val_every_n_epoch
