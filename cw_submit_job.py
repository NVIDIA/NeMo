import nemo_run as run
from nemo.collections import llm
from typing import Optional
import re
from functools import partial
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime


from nemo.collections.llm.recipes import hf_auto_model_for_causal_lm
from nemo import lightning as nl
from nemo.collections.llm import SquadDataModule
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer


DATE_STR = datetime.today().strftime("%m%d")

NEMO_HOME = "/lustre/fsw/portfolios/coreai/users/yudong/github/NeMo"
MEGATRON_HOME = "/lustre/fsw/portfolios/coreai/users/yudong/github/Megatron-LM"
IMAGE = "nvcr.io/nvidia/nemo:24.12"
HF_HOME = "/lustre/fsw/portfolios/coreai/users/yudong/hf_home"
MEGATRON_CACHE = "/lustre/fsw/portfolios/coreai/users/yudong/megatron_cache"
JOB_DIR = "/lustre/fsw/portfolios/coreai/users/yudong/exp/nemorun"
DATA_PATH = "/lustre/fsw/portfolios/coreai/users/yudong/data"


def get_secrets():
    with open("/home/yudong/.bash_secrets") as f:
        text = f.read()
    pattern = r"(\w+)=(\w+)"
    matches = re.findall(pattern, text)
    ret = dict(matches)
    print("Adding secrets: ", ret.keys())
    return ret


def local_executor_torchrun(devices: int = 2) -> run.LocalExecutor:
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(
        ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars
    )
    env_vars.update(get_secrets())
    return executor


def slurm_executor_yudong(
    user: str = "yudong",
    host: str = "cw-dfw-cs-001-login-01",
    remote_job_dir: str = JOB_DIR,
    account: str = "coreai_dlalgo_llm",
    partition: str = "batch",
    nodes: int = 1,
    devices: int = 8,
    time: str = "04:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "/lustre/fsw/coreai_dlalgo_llm/chcui/nvidia+nemo+24.12.sqsh",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (
        user and host and remote_job_dir and account and partition and nodes and devices
    ):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    # CHANGE ME
    mounts = [
        "/lustre:/lustre",
        "/lustre/fsw/portfolios/coreai/users/yudong/:/yudong",
    ]
    # Custom mounts are defined here.
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NEMO_HOME": f"{NEMO_HOME}",
        "HF_HOME": f"{HF_HOME}",
    }
    env_vars.update(get_secrets())
    print(env_vars)
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,  # This is where the results of the run will be stored by default.
            identity="//home/yudong/.ssh/id_ed25519",  # OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,  # eos can't have this
        mem="0",
        exclusive=True,
        gres="gpu:8",  # eos can't have this
        packager=run.Packager(),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def configure_recipe_llama3_8b(
    num_nodes, num_gpus_per_node, peft_scheme, performance_mode, packed_sequence
):
    recipe = llm.llama3_8b.finetune_recipe(
        dir="/chcui/exp/nemorun/checkpoints",  # Path to store checkpoints
        name=f"llama3_8b_{peft_scheme}",
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        peft_scheme=peft_scheme,
        packed_sequence=packed_sequence,
        performance_mode=performance_mode,
    )

    recipe.log.wandb = run.Config(
        WandbLogger,
        project="nemo2",
        name=f"{DATE_STR}-llama3_8b_{peft_scheme}_{'perf' if performance_mode else 'nonperf'}{'_packed' if packed_sequence else ''}",
    )

    return recipe


def configure_recipe_llama32_1b_pretrain(
    num_nodes,
    num_gpus_per_node,
    wandb_project_name=None,
    seq_length=4096,
    global_batch_size=512,
    model_name="meta-llama/Llama-3.2-1B",
):
    model_name = "llama32_1b_pretrain"
    recipe = llm.llama32_1b.pretrain_recipe(
        dir="/yudong/exp/nemorun/checkpoints",  # Path to store checkpoints
        name=model_name,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )
    recipe.trainer.val_check_interval = 100
    # tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(
    #    "meta-llama/Llama-3.2-1B"
    # )
    # recipe.data = run.Config(
    #     SquadHFDataModule,
    #     path_or_dataset="rajpurkar/squad",
    #     split="train",
    #     pad_token_id=tokenizer.tokenizer.eos_token_id,
    #     tokenizer=run.Config(
    #         AutoTokenizer, pretrained_model_name="meta-llama/Llama-3.2-1B"
    #     ),
    #     seq_length=4096,
    #     global_batch_size=512,
    #     micro_batch_size=1,
    # )

    datamodule = run.Config(
        SquadDataModule,
        seq_length=2048,
        global_batch_size=128,
        micro_batch_size=1,
        tokenizer=run.Config(
            AutoTokenizer, pretrained_model_name="meta-llama/Llama-3.2-1B"
        ),
    )
    recipe.data = datamodule

    recipe.trainer.accumulate_grad_batches = (
        global_batch_size / num_gpus_per_node / num_nodes
    )  # Change gradient accumulation steps here

    recipe.log.wandb = run.Config(
        WandbLogger,
        project="nemo2",
        name=f"{DATE_STR}-mcore-{model_name}-{num_nodes}-nodes-{wandb_project_name}",
    )

    return recipe


def configure_recipe_llama31_70b(num_nodes, num_gpus_per_node, peft_scheme):
    recipe = llm.llama31_70b.finetune_recipe(
        dir="/yudong/exp/nemorun/checkpoints",  # Path to store checkpoints
        name=f"llama31_70b_{peft_scheme}",
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        peft_scheme=peft_scheme,
        performance_mode=True,
    )

    recipe.log.wandb = run.Config(
        WandbLogger,
        project="nemo2-squad",
        name=f"nemorun-llama31_70b_{peft_scheme}",
    )

    return recipe


def configure_recipe_llama31_405b(num_nodes, num_gpus_per_node, peft_scheme):
    recipe = llm.llama31_405b.finetune_recipe(
        dir="/yudong/exp/nemorun/checkpoints",  # Path to store checkpoints
        name=f"llama31_405b_{peft_scheme}",
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        peft_scheme=peft_scheme,
        performance_mode=True,
    )

    recipe.log.wandb = run.Config(
        WandbLogger,
        project="nemo2-squad",
        name=f"nemorun-llama31_405b_{peft_scheme}",
    )

    return recipe


def custom_hf_auto_model_for_causal_lm_finetune(
    num_nodes,
    num_gpus_per_node,
    wandb_project_name=None,
    seq_length=4096,
    global_batch_size=512,
    model_name="meta-llama/Llama-3.2-1B",
):
    finetune = hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=model_name,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
    )

    datamodule = run.Config(
        SquadDataModule,
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=1,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=model_name),
    )
    finetune.data = datamodule
    finetune.trainer.accumulate_grad_batches = (
        global_batch_size / num_gpus_per_node / num_nodes
    )  # Change gradient accumulation steps here

    finetune.log.wandb = run.Config(
        WandbLogger,
        project="nemo2",
        name=f"{DATE_STR}-hf-{model_name}-{num_nodes}-nodes-{wandb_project_name}",
    )
    return finetune


def custom_hf_auto_model_for_causal_lm(
    num_nodes,
    num_gpus_per_node,
    wandb_project_name=None,
    seq_length=4096,
    global_batch_size=512,
    model_name="meta-llama/Llama-3.2-1B",
):
    pretrain = hf_auto_model_for_causal_lm.pretrain_recipe(
        model_name=model_name, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node
    )

    pretrain.trainer.max_steps = 10000
    pretrain.trainer.val_check_interval = 100
    pretrain.log.ckpt.save_top_k = -1
    # pretrain.optim = llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)
    # breakpoint()
    # pretrain.data = run.Config(
    #     MockDataModule,
    #     seq_length=seq_length,
    #     global_batch_size=global_batch_size,
    #     micro_batch_size=1,
    # )

    datamodule = run.Config(
        SquadDataModule,
        seq_length=seq_length,
        global_batch_size=global_batch_size,
        micro_batch_size=1,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=model_name),
    )
    pretrain.data = datamodule

    pretrain.trainer.strategy = run.Config(
        nl.FSDP2Strategy,
        data_parallel_size=num_gpus_per_node * num_nodes,
        tensor_parallel_size=1,
    )
    pretrain.trainer.accumulate_grad_batches = (
        global_batch_size / num_gpus_per_node / num_nodes
    )  # Change gradient accumulation steps here

    pretrain.log.wandb = run.Config(
        WandbLogger,
        project="nemo2",
        name=f"{DATE_STR}-hf-{model_name}-{num_nodes}-nodes-{wandb_project_name}",
    )

    return pretrain


recipes = [
    # custom_hf_auto_model_for_causal_lm(1, 2, "nemo2"),
    # custom_hf_auto_model_for_causal_lm(2, 8, "nemo2"),
    # configure_recipe_llama32_1b_pretrain(1, 2),
    # configure_recipe_llama32_1b_pretrain(1, 8),
    # custom_hf_auto_model_for_causal_lm(1, 8, "nemo2"),
    # custom_hf_auto_model_for_causal_lm(2, 8, "nemo2"),
    # custom_hf_auto_model_for_causal_lm(4, 8, "nemo2"),
    # custom_hf_auto_model_for_causal_lm(1, 8, "nemo2", 4096, 32),
    # custom_hf_auto_model_for_causal_lm(1, 8, "nemo2", 4096, 64),
    # custom_hf_auto_model_for_causal_lm(1, 8, "nemo2", 4096, 128),
    # custom_hf_auto_model_for_causal_lm(1, 8, "nemo2", 4096, 256),
    # (
    #     configure_recipe_llama32_1b_pretrain(1, 2, "nemo2"),
    #     "llama32_1b_pretrain_1_node_2_gpu",
    # ),
    (
        custom_hf_auto_model_for_causal_lm(1, 2, "nemo2"),
        "hf_llama32_1b_pretrain_1_node_2_gpu",
    ),
    (
        custom_hf_auto_model_for_causal_lm_finetune(1, 2, "nemo2"),
        "hf_llama32_1b_finetune_1_node_2_gpu",
    ),
    # (configure_recipe_llama32_1b_pretrain(1, 8), "llama32_1b_pretrain_1_node_8_gpu"),
    # (
    #     custom_hf_auto_model_for_causal_lm(1, 8, "nemo2"),
    #     "hf_llama32_1b_pretrain_1_node_8_gpu",
    # ),
    # (
    #     custom_hf_auto_model_for_causal_lm(2, 8, "nemo2"),
    #     "hf_llama32_1b_pretrain_2_node_8_gpu",
    # ),
    # (
    #     custom_hf_auto_model_for_causal_lm(4, 8, "nemo2"),
    #     "hf_llama32_1b_pretrain_4_node_8_gpu",
    # ),
]


def run_local():
    executor = partial(local_executor_torchrun, devices=2)

    with run.Experiment(f"{DATE_STR}-llama3_2_1b_pretrain") as exp:
        for recipe, exp_name in recipes:
            exp.add(recipe, executor=executor(), name=exp_name)
        exp.run(sequential=False, tail_logs=True)


def run_finetuning_on_slurm(**slurm_kwargs):
    num_nodes = 1
    num_gpus_per_node = 8

    executor = partial(
        slurm_executor_yudong,
        # container_image="/lustre/fsw/coreai_dlalgo_llm/chcui/nemo24.12_upgradeTE.sqsh",
        container_image=IMAGE,
        custom_mounts=[
            f"{NEMO_HOME}:/opt/NeMo",
            f"{MEGATRON_HOME}:/opt/megatron-lm",
            f"{HF_HOME}:/opt/hf",
            f"{MEGATRON_CACHE}:/root/.cache/torch/megatron/",
            f"{DATA_PATH}:/data",
        ],
        **slurm_kwargs,
    )

    with run.Experiment(f"{DATE_STR}-squad_llama3_2_1b_pretrain") as exp:
        for recipe, exp_name in zip(recipes, exp_names):
            exp.add(
                recipe, executor=executor(nodes=recipe.trainer.num_nodes), name=exp_name
            )
        exp.run(sequential=False, tail_logs=False)


# Wrap the call in an if __name__ == "__main__": block to work with Python's multiprocessing module.
if __name__ == "__main__":
    print("Update CW codebase")
    breakpoint()
    # run_finetuning_on_slurm()
    run_local()
