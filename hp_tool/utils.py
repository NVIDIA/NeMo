import os
import time
import copy

import yaml
import omegaconf


def _calculate_model_size(
    vocab_size=None, seq_length=None, hidden_size=None, num_layers=None
):
    model_size = (
        12 * num_layers * hidden_size**2
        * (
            1 + (13 / (12 * hidden_size))
            + ((vocab_size + seq_length) / (12 * num_layers * hidden_size))
        )
        / 1e9
    )
    return model_size


def calculate_layers_hs_lr(model_size_in_b, seq_length=2048, vocab_size=51200):
    if model_size_in_b < 0.25:
        hidden_size, att_heads, lr = 768, 12, 6e-4
    elif model_size_in_b < 0.5:
        hidden_size, att_heads, lr = 1024, 16, 3e-4
    elif model_size_in_b < 1:
        hidden_size, att_heads, lr = 1536, 16, 2.5e-4
    elif model_size_in_b < 2:
        hidden_size, att_heads, lr = 2048, 16, 2e-4
    elif model_size_in_b < 3:
        hidden_size, att_heads, lr = 2560, 32, 1.6e-4
    elif model_size_in_b < 4.5:
        hidden_size, att_heads, lr = 3072, 32, 1.4e-4
    elif model_size_in_b < 8:
        hidden_size, att_heads, lr = 4096, 32, 1.2e-4
    elif model_size_in_b < 15:
        hidden_size, att_heads, lr = 5120, 40, 1e-4
    elif model_size_in_b < 25:
        hidden_size, att_heads, lr = 6144, 48, 1e-4
    elif model_size_in_b < 52:
        hidden_size, att_heads, lr = 8192, 64, 0.8e-4
    elif model_size_in_b < 105:
        hidden_size, att_heads, lr = 10240, 80, 0.7e-4
    elif model_size_in_b < 205:
        hidden_size, att_heads, lr = 12288, 96, 0.6e-4
    elif model_size_in_b < 405:
        hidden_size, att_heads, lr = 14336, 128, 0.5e-4
    elif model_size_in_b < 805:
        hidden_size, att_heads, lr = 20480, 128, 0.4e-4
    elif model_size_in_b < 1105:
        hidden_size, att_heads, lr = 24576, 128, 0.3e-4
    else:
        raise ValueError("Model_size must be smaller than 1.1T parameters.")

    # Try powers of 2
    margin = 0.01
    for attempt in range(0, 10):
        for layers in (2**p for p in range(1, 10)):
            out_size = _calculate_model_size(vocab_size, seq_length, hidden_size, layers)
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hidden_size, att_heads, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 16
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(16, 201, 16):
            out_size = _calculate_model_size(vocab_size, seq_length, hidden_size, layers)
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hidden_size, att_heads, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 2
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(2, 201, 2):
            out_size = _calculate_model_size(vocab_size, seq_length, hidden_size, layers)
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hidden_size, att_heads, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try any valid number
    margin = 0.01
    for attempt in range(0, 10):
        for layers in range(1, 200):
            out_size = _calculate_model_size(vocab_size, seq_length, hidden_size, layers)
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hidden_size, att_heads, lr
        margin += 0.01  # Double margin of acceptable model sizes.
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
    run:
      name: 126m
      results_dir: ${base_results_dir}/${.name}
      time_limit: "1-12:00:00"
      dependency: "singleton"

    trainer:
      gpus: 8
      num_nodes: 8
      accelerator: ddp
      precision: bf16
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
      explicit_log_dir: ${training.run.results_dir}
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
        model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
      log_step_timing: True
      step_timing_kwargs:
        sync_cuda: True
        buffer_size: ${multiply:100, ${training.trainer.accumulate_grad_batches}}

    model:
      micro_batch_size: 4
      global_batch_size: 256
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
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
      persist_layer_norm: True
      gradient_as_bucket_view: True
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
      hysteresis: 2
      fp32_residual_connection: False # Move residual connections to fp32
      fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16
      megatron_amp_O2: True

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
    return base_cfg


def modify_cfg(base_cfg, act, tp, pp, mbs, max_minutes, max_pp):
    new_cfg = copy.deepcopy(base_cfg)
    new_cfg["model"]["activations_checkpoint_num_layers"] = act
    new_cfg["model"]["tensor_model_parallel_size"] = tp
    new_cfg["model"]["pipeline_model_parallel_size"] = pp
    new_cfg["model"]["micro_batch_size"] = mbs
    att_heads = new_cfg["model"]["num_attention_heads"]
    num_layers = new_cfg["model"]["num_layers"]

    # gbs = mbs * num_gpus * accumulate_grad_batches / (tp * pp)
    num_gpus = new_cfg["trainer"]["num_nodes"] * new_cfg["trainer"]["gpus"]
    gbs = new_cfg["model"]["global_batch_size"]

    mod_gbs = gbs % (mbs * num_gpus / (tp * pp))
    mod_att_heads = att_heads % tp
    mod_layers = num_layers % pp
    if mod_gbs == 0 and mod_att_heads == 0 and mod_layers == 0:
        # Valid config
        new_cfg["trainer"]["num_nodes"] = max_pp  # Necessary for short single-node test.
        days = max_minutes // 3600
        hours = (max_minutes % 3600) // 60
        mins = (max_minutes % 3600) % 60
        new_cfg["run"]["time_limit"] = f"{days}-{hours}:{mins}:00"
        new_cfg["run"]["name"] = \
                f"{new_cfg['run']['name']}_tp_{tp}_pp_{pp}_mbs_{mbs}_act_ckpt_{act}"
        print(
            f"Valid config: GBS={gbs}, MBS={mbs}, TP={tp}, PP={pp}, act_ckpt_layers={act}. Adding to directory."
        )
        return new_cfg
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


def convert_to_cli(cfg):
    result = ""
    for k, v in cfg.items():
        if isinstance(v, omegaconf.dictconfig.DictConfig):
            output = convert_to_cli(v).split(" ")
            result += " ".join([f"{k}.{x}" for x in output if x != ""]) + " "
        elif isinstance(v, omegaconf.listconfig.ListConfig):
            if k == "data_prefix":
                if v is None:
                    v = "null"
                else:
                    v = [x for x in v]  # Needed because of lazy omegaconf interpolation.
            result += f"{k}={str(v).replace(' ', '')} "
        elif isinstance(v, str) and "{" in v:
            continue
        elif k in ["splits_string", "file_numbers", "languages"]:
            result += f"{k}=\\'{v}\\' "
        elif k == "checkpoint_name":
            v = v.replace('=', '\=')
            result += f"{k}=\'{v}\' "
        else:
            result += f"{k}={convert_to_null(v)} "
    return result

def convert_to_null(val):
    if val is None:
        return "null"
    return val

