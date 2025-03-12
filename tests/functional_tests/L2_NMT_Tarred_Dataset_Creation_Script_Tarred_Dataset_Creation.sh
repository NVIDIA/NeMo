cd examples/nlp/machine_translation && \
python create_tarred_parallel_dataset.py \
--src_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
--tgt_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
--out_dir $PWD/out_dir \
--encoder_tokenizer_vocab_size=2000 \
--decoder_tokenizer_vocab_size=2000 \
--tokens_in_batch=1000 \
--lines_per_dataset_fragment=500 \
--num_batches_per_tarfile=10 \
--n_preproc_jobs=2
