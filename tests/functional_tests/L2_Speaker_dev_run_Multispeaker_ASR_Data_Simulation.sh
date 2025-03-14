coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tools/speech_data_simulator/multispeaker_simulator.py \
    --config-path=conf --config-name=data_simulator.yaml \
    data_simulator.random_seed=42 \
    data_simulator.manifest_filepath=/home/TestData/LibriSpeechShort/dev-clean-align-short.json \
    data_simulator.outputs.output_dir=/tmp/test_simulator \
    data_simulator.session_config.num_sessions=2 \
    data_simulator.session_config.session_length=60
