import os
import time
import copy

import yaml


def _calculate_model_size(
    vocab_size=None, seq_length=None, hidden_size=None, num_layers=None
):
    model_size = (
        12
        * num_layers
        * hidden_size**2
        * (
            1
            + (13 / (12 * hidden_size))
            + ((vocab_size + seq_length) / (12 * num_layers * hidden_size))
        )
        / 1e9
    )
    return model_size


def calculate_num_layers_hidden_size_learning_rate(
    model_size, seq_length=2048, vocab_size=51200
):
    if model_size < 0.25:
        hidden_size, att_heads, lr = 768, 12, 6e-4
    elif model_size < 0.5:
        hidden_size, att_heads, lr = 1024, 16, 3e-4
    elif model_size < 1:
        hidden_size, att_heads, lr = 1536, 16, 2.5e-4
    elif model_size < 2:
        hidden_size, att_heads, lr = 2048, 16, 2e-4
    elif model_size < 3:
        hidden_size, att_heads, lr = 2560, 32, 1.6e-4
    elif model_size < 4.5:
        hidden_size, att_heads, lr = 3072, 32, 1.4e-4
    elif model_size < 8:
        hidden_size, att_heads, lr = 4096, 32, 1.2e-4
    elif model_size < 15:
        hidden_size, att_heads, lr = 5120, 40, 1e-4
    elif model_size < 25:
        hidden_size, att_heads, lr = 6144, 48, 1e-4
    else:
        hidden_size, att_heads, lr = 12288, 96, 0.6e-4

    margin = 0.02
    for attempt in range(0, 10):
        for num_layers in range(1, 100):
            out_model_size = _calculate_model_size(
                vocab_size, seq_length, hidden_size, num_layers
            )
            if (
                model_size * (1.0 - margin)
                < out_model_size
                < model_size * (1.0 + margin)
            ):
                return num_layers, hidden_size, att_heads, lr
        margin += 0.03
    raise Exception("Number of layers not found, config is not possible.")


def estimate_training_time(
    model_size=None, num_tokens=None, num_gpus=None, tflops_per_gpu=None
):
    training_time = (8 * num_tokens * 1e9 * model_size * 1e9) / (
        num_gpus * tflops_per_gpu * 1e12 * 3600 * 24
    )
    return training_time


def generic_base_config(cfg):
    inp = """\
    slurm:
      partition: ???
      account: "null"
      time_limit: "1-00:00:00"
      nodes: 8
      exclusive: True
      mem: 0
      overcommit: True
      ntasks_per_node: 8
      gpus_per_task: "null"
      dependency: "singleton"
      job_name: "bignlp-gpt3:126m"

    run:
      name: "126m"
      log_dir: ${bignlp_path}/train_config_generator/logs/126m

    trainer:
      gpus: ${training.slurm.ntasks_per_node}
      num_nodes: ${training.slurm.nodes}
      accelerator: ddp
      precision: 16
      amp_backend: native
      logger: False # logger provided by exp_manager
      checkpoint_callback: False
      replace_sampler_ddp: False
      max_epochs: "null"
      max_steps: 600000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
      max_time: 01:12:00:00 # days:hours:minutes:seconds
      log_every_n_steps: 1
      val_check_interval: ${multiply:2000, ${.accumulate_grad_batches}}
      limit_val_batches: ${multiply:50, ${.accumulate_grad_batches}}
      limit_test_batches: ${multiply:50, ${.accumulate_grad_batches}}
      accumulate_grad_batches: 1
      gradient_clip_val: 1.0

    exp_manager:
      explicit_log_dir: ${training.run.log_dir}
      exp_dir: "null"
      name: megatron_gpt
      create_wandb_logger: False
      wandb_logger_kwargs:
        project: nemo_gpt_pretraining
        name: dev
      resume_if_exists: True
      resume_ignore_no_checkpoint: True
      create_checkpoint_callback: True
      checkpoint_callback_params:
        monitor: val_loss
        save_top_k: 10
        mode: min
        always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
        filename: 'megatron_gpt--{val_loss:.2f}-{step}-{consumed_samples}'
        model_parallel_size: ${training.model.tensor_model_parallel_size}
      log_step_timing: True
      step_timing_kwargs:
        sync_cuda: True
        buffer_size: ${multiply:100, ${training.trainer.accumulate_grad_batches}}


    model:
      micro_batch_size: 4
      tensor_model_parallel_size: 1
      encoder_seq_length: 2048
      max_position_embeddings: 2048
      num_layers: 12
      hidden_size: 768
      ffn_hidden_size: ${multiply:4, ${.hidden_size}}  # Transformer FFN hidden size. 4 * hidden_size.
      num_attention_heads: 12
      init_method_std: 0.023  # Standard deviation of the zero mean normal distribution used for weight initialization.')
      hidden_dropout: 0.1  # Dropout probability for hidden state transformer.
      kv_channels: "null"  # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if "null"
      apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
      layernorm_epsilon: 1e-5
      make_vocab_size_divisible_by: 128 # Pad the vocab size to be divisible by this value for computation efficiency.
      pre_process: True # add embedding
      post_process: True # add pooler
      activations_checkpoint_method: block
      activations_checkpoint_num_layers: 0

      tokenizer:
        library: 'megatron'
        type: 'GPT2BPETokenizer'
        model: "null"
        vocab_file: ${bignlp_path}/data_preparation/bpe/vocab.json
        merge_file: ${bignlp_path}/data_preparation/bpe/merges.txt

      native_amp_init_scale: 4294967296 # 2 ** 32
      native_amp_growth_interval: 1000
      fused_fp16: True # False if using fp32 or bf16
      fused_bf16: False # True if using bf16
      fp32_residual_connection: False # Move residual connections to fp32
      fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16

      seed: 1234
      use_cpu_initialization: False # Init weights on the CPU (slow for large models)
      onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.

      optim:
        name: fused_adam
        lr: 6e-4
        weight_decay: 0.1 
        betas: 
        - 0.9
        - 0.95
        sched:
          name: CosineAnnealing
          warmup_steps: 636
          constant_steps: 100000
          min_lr: 6e-5

      data:
        data_impl: mmap
        splits_string: "90,5,5"
        seq_length: 2048
        skip_warmup: True
        num_workers: 2
        dataloader_type: single # cyclic
        reset_position_ids: False # Reset position ids after end-of-document token
        reset_attention_mask: False # Reset attention mask after end-of-document token
        eod_mask_loss: False # Mask loss for the end of document tokens
        data_prefix: # Should be weight path weight path... for a blended dataset
          - 1.0
          - ${data_dir}/my-gpt3_00_text_document
    """
    base_cfg = yaml.safe_load(inp)
    base_cfg["slurm"] = dict(cfg["slurm"])
    return base_cfg


def modify_cfg(base_cfg, gbs, act_layers, tp, mbs, max_mins, model_size):
    new_cfg = copy.deepcopy(base_cfg)
    new_cfg["model"]["activations_checkpoint_num_layers"] = act_layers
    new_cfg["model"]["tensor_model_parallel_size"] = tp
    new_cfg["model"]["micro_batch_size"] = mbs
    att_heads = new_cfg["model"]["num_attention_heads"]

    # gbs = mbs * num_gpus * accumulate_grad_batches / tp
    num_gpus = new_cfg["slurm"]["nodes"] * new_cfg["slurm"]["ntasks_per_node"]

    # Set accumulate_grad_batches accordingly
    mod_gbs = gbs % (mbs * num_gpus / tp)
    mod_att_heads = att_heads % tp
    if mod_gbs == 0 and mod_att_heads == 0:
        # Valid config
        accum = gbs / (mbs * num_gpus / tp)
        new_cfg["trainer"]["accumulate_grad_batches"] = int(accum)
        new_cfg["slurm"]["nodes"] = 1  # Necessary for short single-node test.
        days = max_mins // 3600
        hours = (max_mins % 3600) // 60
        mins = (max_mins % 3600) % 60
        new_cfg["slurm"]["time_limit"] = f"{days}-{hours}:{mins}:00"
        new_cfg["slurm"][
            "job_name"
        ] = f"{new_cfg['slurm']['job_name']}_tp_{tp}_mbs_{mbs}_act_ckpt_{act_layers}"
        new_cfg["run"][
            "log_dir"
        ] = f"{new_cfg['run']['log_dir']}/tp_{tp}_mbs_{mbs}_act_ckpt_{act_layers}"
        print(
            f"I: Valid config: GBS={gbs}, MBS={mbs}, TP={tp}, num_gpus={num_gpus}, act_ckpt_layers={act_layers}. Adding to directory."
        )
        return new_cfg
    elif mod_gbs != 0:
        print(
            f"W: Invalid config: GBS={gbs}, MBS={mbs}, TP={tp}, num_gpus={num_gpus}, act_ckpt_layers={act_layers}. GBS must be a multiple of MBS * data_parallelism."
        )
    elif mod_att_heads != 0:
        print(
            f"W: Invalid config: TP={tp}, num_attention_heads={att_heads}. num_attention_heads must be a multiple of TP."
        )
    return None


def create_slurm_file(
    new_script_path,
    train_cmd,
    job_name,
    flags="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    mem=0,
    overcommit=True,
    nodes=1,
    ntasks_per_node=8,
    gpus_per_task=1,
    partition="batch",
    account=None,
):
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines(f"#SBATCH --nodes={nodes}\n")
        f.writelines(f"#SBATCH --ntasks-per-node={ntasks_per_node}\n")
        if gpus_per_task is not None:
            f.writelines(f"#SBATCH --gpus-per-task={gpus_per_task}\n")
        if dependency is not None:
            if dependency != "singleton":
                dependency = f"afterany:{dependency}"
            f.writelines(f"#SBATCH --dependency={dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        if account is not None:
            f.writelines(f"#SBATCH -A {account}\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        f.writelines(f"#SBATCH --mem={mem}\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        if overcommit:
            f.writelines("#SBATCH --overcommit\n")
        f.writelines(f"#SBATCH --time={time}\n\n")
        f.writelines(f'srun {flags} sh -c "{train_cmd}"\n\n')
        f.writelines("set +x\n")
