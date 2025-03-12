cd examples/tts/g2p && \
    TIME=`date +"%Y-%m-%d-%T"` && OUTPUT_DIR_CONFORMER=output_ctc_${TIME} && \
    python g2p_train_and_evaluate.py \
        train_manifest=/home/TestData/g2p/g2p.json \
        validation_manifest=/home/TestData/g2p/g2p.json \
        model.test_ds.manifest_filepath=/home/TestData/g2p/g2p.json \
        model.tokenizer.dir=/home/TestData/g2p/tokenizer_spe_unigram_v512 \
        trainer.max_epochs=1 \
        model.max_source_len=64 \
        trainer.devices=1 \
        do_training=True \
        do_testing=True \
        exp_manager.exp_dir=${OUTPUT_DIR_CONFORMER} \
        +exp_manager.use_datetime_version=False\
        +exp_manager.version=test \
        --config-name=g2p_conformer_ctc && \
    python g2p_inference.py \
        pretrained_model=${OUTPUT_DIR_CONFORMER}/G2P-Conformer-CTC/test/checkpoints/G2P-Conformer-CTC.nemo \
        manifest_filepath=/home/TestData/g2p/g2p.json \
        phoneme_field=text
