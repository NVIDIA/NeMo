cd examples/tts/g2p &&
    TIME=$(date +"%Y-%m-%d-%T") && OUTPUT_DIR=output_${TIME} &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo g2p_heteronym_classification_train_and_evaluate.py \
        train_manifest=/home/TestData/g2p/manifest.json \
        validation_manifest=/home/TestData/g2p/manifest.json \
        test_manifest=/home/TestData/g2p/manifest.json \
        model.wordids=/home/TestData/g2p/wordids.tsv \
        trainer.max_epochs=1 \
        model.max_seq_length=64 \
        do_training=True \
        do_testing=True \
        exp_manager.exp_dir=${OUTPUT_DIR} \
        +exp_manager.use_datetime_version=False +exp_manager.version=test &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo g2p_heteronym_classification_inference.py \
        manifest=/home/TestData/g2p/manifest.json \
        pretrained_model=${OUTPUT_DIR}/HeteronymClassification/test/checkpoints/HeteronymClassification.nemo \
        output_manifest=preds.json
