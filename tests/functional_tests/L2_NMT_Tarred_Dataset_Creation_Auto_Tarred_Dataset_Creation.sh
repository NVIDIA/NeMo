cd examples/nlp/machine_translation && \
python enc_dec_nmt.py \
--config-path=conf \
--config-name=aayn_base \
do_training=false \
model.preproc_out_dir=$PWD/preproc_out_dir \
model.train_ds.use_tarred_dataset=true \
model.train_ds.n_preproc_jobs=2 \
model.train_ds.lines_per_dataset_fragment=500 \
model.train_ds.num_batches_per_tarfile=10 \
model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.encoder_tokenizer.vocab_size=2000 \
model.decoder_tokenizer.vocab_size=2000 \
~model.test_ds \
trainer.devices=1 \
trainer.accelerator="gpu" \
+trainer.fast_dev_run=true \
exp_manager=null
