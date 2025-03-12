python examples/tts/waveglow.py \
train_dataset=/home/TestData/an4_dataset/an4_train.json \
validation_datasets=/home/TestData/an4_dataset/an4_val.json \
trainer.devices="[0]" \
+trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
trainer.strategy=auto \
model.train_ds.dataloader_params.batch_size=4 \
model.train_ds.dataloader_params.num_workers=0 \
model.validation_ds.dataloader_params.batch_size=4 \
model.validation_ds.dataloader_params.num_workers=0 \
model.waveglow.n_flows=4 \
model.waveglow.n_wn_layers=2 \
model.waveglow.n_wn_channels=32 \
~trainer.check_val_every_n_epoch
