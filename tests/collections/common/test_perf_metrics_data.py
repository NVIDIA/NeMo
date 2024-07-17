
LLAMA2_CFG_STR = """
    run:
        name: train_llama2_7b_tp1_pp1_FP8_1node_15steps
        results_dir: "results"
    trainer:
        num_nodes: 1
        devices: 8
        accelerator: gpu
        precision: bf16
    exp_manager:
        explicit_log_dir: "results/logs"
    model:
        micro_batch_size: 1
        global_batch_size: 128
        tensor_model_parallel_size: 1
        pipeline_model_parallel_size: 1
        virtual_pipeline_model_parallel_size: null
        encoder_seq_length: 4096
        max_position_embeddings: 4096
        num_layers: 32
        hidden_size: 4096
        ffn_hidden_size: 11008
        num_attention_heads: 32
"""

NEMOTRON_CFG_STR = """
    run:
      name: train_nemotron-8b-tp2-pp1-FP8-8node-20steps
      results_dir: "results"
      time_limit: 00:30:00
      dependency: singleton
    trainer:
      num_nodes: 8
      devices: 8
      accelerator: gpu
      precision: bf16
    exp_manager:
      explicit_log_dir: null
    model:
      micro_batch_size: 4
      global_batch_size: 256
      context_parallel_size: 1
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 1
      virtual_pipeline_model_parallel_size: null
      encoder_seq_length: 4096
      max_position_embeddings: 4096
      num_layers: 32
      hidden_size: 4096
      ffn_hidden_size: 16384
      num_attention_heads: 32
      fp8: true
"""

UNSUPPORTED_MODEL_CFG_STR = """
    run:
        name: unsupported_model
    trainer:
        num_nodes: 1
        devices: 8
        accelerator: gpu
        precision: bf64
    exp_manager:
        explicit_log_dir: null
    model:
        micro_batch_size: 1
        global_batch_size: 1
        tensor_model_parallel_size: 1
        pipeline_model_parallel_size: 1
        virtual_pipeline_model_parallel_size: 1
        encoder_seq_length: 1
        max_position_embeddings: 1
        num_layers: 1
        hidden_size: 1
        ffn_hidden_size: 1
        num_attention_heads: 1
"""