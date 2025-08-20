import nemo_run as run

from nemo.collections import llm
from nemo.collections.llm.recipes import qwen3_32b
from nemo.collections.llm.recipes.finetune_default import nemo_resume
from nemo import lightning as nl
import os

JOB_DIR = "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo"
SLURM_ACCOUNT = os.environ.get("SLURM_ACCOUNT", "coreai_dlalgo_genai")
SLURM_PARTITION = os.environ.get("SLURM_PARTITION", "interactive")
# BASE_IMAGE = os.environ.get("BASE_IMAGE", "nvcr.io#nvidia/nemo:25.07")
BASE_IMAGE = os.environ.get("BASE_IMAGE", "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/nemo-25.07-dev.sqsh")
NEMO_DIR = "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo"

def custom_qwen3_32b():
    # pretrain_recipe defaults to mock dataset, with the qwen tokenizer.
    recipe = qwen3_32b.pretrain_recipe(
        name='HEHE',
        dir=os.path.join(JOB_DIR, "checkpoints"),
        pipeline_parallelism=1,
        num_nodes=1,
        max_steps = 3,
        num_gpus_per_node=8,
        global_batch_size=4,
        micro_batch_size=1,
        seq_length=512,   # Shorten it for testing.
    )
    recipe.resume = nemo_resume("Qwen/Qwen3-32B")  # Defaults to f"nemo://{model_id}""
    recipe.log.name = 'qwen-3-32b-cpt'
    recipe.log.ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=1,
        filename="{model_name}--{val_loss:.2f}-{step}-{consumed_samples}",
    )
    recipe.trainer.max_steps = 4
    recipe.trainer.strategy.ckpt_async_save = True
    recipe.trainer.val_check_interval = 4
    ##--------------------------------------------------------------------------

    print(recipe)
    return recipe

if __name__ == "__main__":
    ## run.cli.main(llm.finetune, default_factory=custom_llama32_1b)

    recipe = custom_qwen3_32b()
    local_tunnel = run.LocalTunnel(job_dir=JOB_DIR)
    ##run.run(recipe, direct=True)

    ## NOTE: run.run(recipe, direct=True) is equivalent to:
    # from nemo_run.run.task import direct_run_fn
    # direct_run_fn(recipe)
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/lustre/fsw/portfolios/coreai/users/zhiyul/hf",
        "NEMO_MODELS_CACHE": "/lustre/fsw/portfolios/coreai/users/zhiyul/nemo",   # cache path
    }
    executor = None
    use_slurm = False
    if use_slurm:
        executor = run.SlurmExecutor(
            account=SLURM_ACCOUNT,
            partition=SLURM_PARTITION,
            nodes=1,
            gpus_per_node=8,
            ntasks_per_node=8,  # Typically 1 task per GPU
            time="02:00:00" if SLURM_PARTITION == "interactive" else "04:00:00",
            container_image=BASE_IMAGE,
            job_name_prefix="zhiyul_nemo_pretrain-",
            # launcher="torchrun" is often used for multi-gpu/multi-node jobs
            # When a launcher like Torchrun is used, `ntasks_per_node` is automatically
            # set to 1 and the value is passed to torchrun's nproc_per_node argument.
            launcher="torchrun",
            container_mounts=["/lustre:/lustre", f"{NEMO_DIR}:/opt/NeMo"],
            env_vars=env_vars,
        )
    else:
        executor = run.LocalExecutor(
            nodes=1,
            ntasks_per_node=8,  # Typically 1 task per GPU
            env_vars=env_vars,
            launcher="torchrun",
        )

    run.run(
        recipe,
        executor=executor,
        name="multinode_qwen3_32b_pretraining"
    )
