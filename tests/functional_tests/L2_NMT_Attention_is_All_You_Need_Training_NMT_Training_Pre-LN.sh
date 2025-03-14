cd examples/nlp/machine_translation &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo enc_dec_nmt.py \
        --config-path=conf \
        --config-name=aayn_base \
        do_testing=true \
        model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
        model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
        model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
        model.encoder.pre_ln=true \
        model.decoder.pre_ln=true \
        trainer.devices=1 \
        trainer.accelerator="gpu" \
        +trainer.fast_dev_run=true \
        +trainer.limit_test_batches=2 \
        exp_manager=null
