from nemo.collections.llm.tools.auto_configurator import get_results

get_results(
    training_logs="/home/llama3_8b",
    path_to_save="/home/test_auto_conf",
    model_name="llama",
    vocab_size=32000,
    model_size=8,
    global_batch_size=1024,
    model_version=3,
    num_nodes=16,
    seq_length=8192,
)
