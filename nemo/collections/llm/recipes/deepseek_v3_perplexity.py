import argparse

import nemo_run as run

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.recipes.finetune_default import nemo_resume
from nemo.collections.llm.recipes.log.default import default_log, wandb_logger


def dsr1_full_finetune(name: str):
    dir = "/traindata/jianqun/runs/NeMo/dsr1_sft"
    steps_per_epoch = 6489
    n_epochs = 1
    num_nodes = 16
    gas = 4

    recipe = llm.deepseek_v3.finetune_recipe(
        name=name,
        dir=dir,
        num_nodes=num_nodes,
        peft_scheme='none',  # 'lora', 'none'
        packed_sequence=False,
    )
    recipe.resume = nemo_resume("/traindata/nemo_models/base/dsr1-v2-eugen")
    # Required to ignore fp8 states, otherwise checkpoint loading will fail.
    recipe.trainer.strategy.ckpt_load_strictness = "log_all"

    # Dataset.
    recipe.data = run.Config(
        llm.FineTuningDataModule,
        dataset_root="/traindata/jianqun/data/censorship/nemo",
        tokenizer=run.Config(AutoTokenizer, "/traindata/eugen/DeepSeek-R1-bf16/"),
        seq_length=16384,
        micro_batch_size=1,
        global_batch_size=num_nodes * gas,
        dataset_kwargs={"chat": True}
    )

#    recipe.data = run.Config(
#        llm.HFDatasetDataModule,
#        path_or_dataset="json",
#        tokenizer_path="/traindata/eugen/DeepSeek-R1-bf16/",
#        data_files="/traindata/jianqun/data/censorship/nemo/training.jsonl",
#        split="train",
#        seq_length=16384,
#        micro_batch_size=1,
#        global_batch_size=num_nodes * gas,
#        use_mcore_sampler=True,
#    )

    # Memory optimizations.
    recipe.trainer.strategy.expert_model_parallel_size = num_nodes
    recipe.trainer.strategy.pipeline_model_parallel_size = 8
    recipe.model.config.recompute_granularity = "full" # 'selective' or 'full'
    recipe.model.config.recompute_method = "uniform" # 'uniform', 'block', not used with 'selective'
    recipe.model.config.recompute_num_layers = 1

    recipe.model.config.moe_aux_loss_coeff = 0.0 # disable aux loss

    # Training configs.
    recipe.optim.config.lr = 5e-6
    recipe.optim.lr_scheduler.warmup_steps = 30
    recipe.trainer.max_steps = steps_per_epoch * n_epochs
    recipe.trainer.limit_val_batches = 0.0  # Disables validation.
    recipe.trainer.limit_test_batches = 0.0  # Disables testing.

    # Logging configs.
    recipe.trainer.log_every_n_steps = 1
    recipe.log = default_log(
        dir=dir,
        name=name,
        wandb_logger=wandb_logger(project="deepseek-r1-sft", name=name),
    )
    recipe.log.ckpt = run.Config(
        nl.ModelCheckpoint,
        monitor="reduced_train_loss",
        save_last=True,
        #save_weights_only=True,
        save_top_k=2,
        every_n_train_steps=200,
        filename="{model_name}-{step}-{consumed_samples}",
    )

    return recipe, num_nodes


def dsr1_peft(name: str):
    dir = "/traindata/eugen/runs/NeMo/dsr1_sft"
    num_nodes = 6
    gas = 2

    recipe = llm.deepseek_v3.finetune_recipe(
        name=name,
        dir=dir,
        num_nodes=num_nodes,
        peft_scheme='lora',  # 'lora', 'none'
        packed_sequence=False,
    )
    recipe.resume = nemo_resume("/traindata/nemo_models/base/dsr1-v2-eugen")
    # Required to ignore fp8 states, otherwise checkpoint loading will fail.
    recipe.trainer.strategy.ckpt_load_strictness = "log_all"

    recipe.data = run.Config(
        llm.HFDatasetDataModule,
        path_or_dataset="json",
        tokenizer_path="/traindata/eugen/DeepSeek-R1-bf16/",
        data_files="/traindata/jianqun/data/censorship/nemo/training.jsonl",
        split="train",
        seq_length=16384,
        micro_batch_size=1,
        global_batch_size=16 * gas,
        use_mcore_sampler=True,
    )

    recipe.trainer.max_steps = 100
    recipe.trainer.limit_val_batches = 0.0  # Disables validation.
    recipe.trainer.limit_test_batches = 0.0  # Disables testing.

    return recipe, num_nodes


def slurm_executor(num_nodes: int) -> run.SlurmExecutor:
    mounts = [
        "/traindata:/traindata",
        "/traindata/eugen/NeMo:/opt/NeMo",
        "/traindata/eugen/Megatron-LM:/opt/megatron-lm",
    ]

    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account="root",
        partition="dev",
        nodes=num_nodes,
        ntasks_per_node=8,
        gpus_per_node=8,
        mem="0",
        exclusive=True,
        gres="gpu:8",
        packager=run.Packager(),
    )

    executor.container_image = "/traindata/enroot_images/nemofw_r1.sqsh/nemofw_r1.sqsh"
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = 0
    executor.time = "48:00:00"

    return executor


def run_recipe(name: str):
    recipe, num_nodes = dsr1_full_finetune(name=name)
    #recipe, num_nodes = dsr1_peft(name=name)
    executor = slurm_executor(num_nodes)

    run.run(recipe, executor=executor, detach=False, name="dsr1_sft_jianqun800k_rerun")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    run_recipe(args.name)
