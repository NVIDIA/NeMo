python examples/tts/fastpitch_finetune.py --config-name=fastpitch_align_v1.05.yaml \
  train_dataset=data/PROJECT-907dbaf1/data.json \
  validation_datasets=data/PROJECT-907dbaf1/data.json \
  sup_data_path=.data/PROJECT-907dbaf1/data_cache \
  +init_from_nemo_model=saved_models/fastpitch_2/checkpoints/FastPitch.nemo \
  +trainer.max_steps=1000 ~trainer.max_epochs \
  trainer.check_val_every_n_epoch=5 \
  model.train_ds.dataloader_params.batch_size=8 model.validation_ds.dataloader_params.batch_size=8 \
  model.n_speakers=1 model.optim.lr=2e-4 \
  ~model.optim.sched model.optim.name=adam trainer.devices=1 trainer.strategy=null \
  trainer.accelerator=cpu
