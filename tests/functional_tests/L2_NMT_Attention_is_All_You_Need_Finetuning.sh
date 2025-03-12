cd examples/nlp/machine_translation && \
python enc_dec_nmt_finetune.py \
model_path=/home/TestData/nlp/nmt/toy_data/enes_v16k_s100k_6x6.nemo \
trainer.devices=1 \
~trainer.max_epochs \
model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
+trainer.val_check_interval=10 \
+trainer.limit_val_batches=1 \
+trainer.limit_test_batches=1 \
+trainer.max_steps=10 \
+exp_manager.exp_dir=examples/nlp/machine_translation/nmt_finetune \
+exp_manager.create_checkpoint_callback=True \
+exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
+exp_manager.checkpoint_callback_params.mode=max \
+exp_manager.checkpoint_callback_params.save_best_model=true
