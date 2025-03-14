coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/test_hf_import.py \
  --hf_model /home/TestData/nlp/megatron_llama/llama-ci-hf \
  --output_path /tmp/nemo2_ckpt

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/setup/data/create_sample_lambada.py \
  --output_file /tmp/lambada.json

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/export/nemo_export.py \
  --model_name test \
  --model_type llama \
  --checkpoint_dir /tmp/nemo2_ckpt \
  --min_tps 1 \
  --in_framework True \
  --test_deployment True \
  --run_accuracy True \
  --test_data_path /tmp/lambada.json \
  --accuracy_threshold 0.0 \
  --debug
