mkdir examples/llm/auto_configurator/auto_conf_logs

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/llm/auto_configurator/auto_config.py \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
    --run_number=1

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/llm/auto_configurator/auto_config.py \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
    --run_number=2

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/llm/auto_configurator/auto_config.py \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
    --run_number=3

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/llm/auto_configurator/auto_config.py \
    --log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
    --get_results
