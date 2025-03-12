mkdir examples/llm/auto_configurator/auto_conf_logs

python examples/llm/auto_configurator/auto_config.py \
--log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
--run_number=1

python examples/llm/auto_configurator/auto_config.py \
--log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
--run_number=2

python examples/llm/auto_configurator/auto_config.py \
--log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
--run_number=3

python examples/llm/auto_configurator/auto_config.py \
--log_dir=/workspace/examples/llm/auto_configurator/auto_conf_logs \
--get_results
