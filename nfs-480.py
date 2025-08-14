import nemo_run as run
import os
from nemo.collections import llm

if __name__ == "__main__":
    # 1. Configure your function/task
    # This example uses a pre-defined recipe from NeMo.
    # The `num_nodes` and `num_gpus_per_node` here should align with the executor's configuration.
    JOB_DIR = "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo"
    SLURM_ACCOUNT = os.environ.get("SLURM_ACCOUNT", "coreai_dlalgo_genai")
    BASE_IMAGE = os.environ.get("BASE_IMAGE", "nvcr.io#nvidia/nemo:25.07")
    # BASE_IMAGE = os.environ.get("BASE_IMAGE", "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/nemo-25.07-dev.sqsh")
    os.environ["HF_HOME"] = "/lustre/fsw/portfolios/coreai/users/zhiyul/hf"

    # model = "llama3-8b"
    model = "Qwen3_235B_A22B"

    if model == "llama3-8b":
        # It is important that the configuration of the task matches the resources requested by the executor.
        llm.import_ckpt(model=llm.LlamaModel(llm.Llama31Config8B(apply_rope_fusion=False)), source='hf://meta-llama/Llama-3.1-8B-Instruct')
        NUM_NODES = 1
        GPUS_PER_NODE = 8
        SLURM_PARTITION = os.environ.get("SLURM_PARTITION", "interactive")
        partial_func = llm.llama31_8b.finetune_recipe(
            name="llama3-8b-multinode-finetune",
            dir=JOB_DIR,
            num_nodes=NUM_NODES,
            num_gpus_per_node=GPUS_PER_NODE,
        )
    elif model == "Qwen3_235B_A22B":
        NUM_NODES = 16
        GPUS_PER_NODE = 8
        SLURM_PARTITION = os.environ.get("SLURM_PARTITION", "batch")
        partial_func = llm.qwen3_235b_a22b.finetune_recipe(
            name="Qwen3-235B-A22B-multinode-finetune",
            num_nodes=NUM_NODES,
            num_gpus_per_node=GPUS_PER_NODE,
        )

    partial_func.trainer.max_steps = 5
    partial_func.trainer.strategy.ckpt_async_save = True
    partial_func.trainer.val_check_interval = 5

    # Using LocalTunnel as we are on a SLURM login node.
    # For remote submission, use run.SSHTunnel instead.
    local_tunnel = run.LocalTunnel(job_dir=JOB_DIR)

    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/lustre/fsw/portfolios/coreai/users/zhiyul/hf",
        "NEMO_MODELS_CACHE": "/lustre/fsw/portfolios/coreai/users/zhiyul/nemo",
        "HUGGINGFACE_TOKEN": "<HF_TOKEN>",   # TODO: replace with your own token
        "TRITON_CACHE_DIR": f"/tmp/triton_cache_{os.environ.get('SLURM_NODEID')}",
        "TRITON_DISABLE_CACHE": "1",
        "TRITON_CACHE_DISABLE": "1",
    }

    slurm_executor = run.SlurmExecutor(
        account=SLURM_ACCOUNT,
        partition=SLURM_PARTITION,
        nodes=NUM_NODES,
        gpus_per_node=GPUS_PER_NODE,
        ntasks_per_node=GPUS_PER_NODE,  # Typically 1 task per GPU
        time="02:00:00" if SLURM_PARTITION == "interactive" else "04:00:00",
        container_image=BASE_IMAGE,
        job_name_prefix="zhiyul_nemo_finetune-",
        # launcher="torchrun" is often used for multi-gpu/multi-node jobs
        # When a launcher like Torchrun is used, `ntasks_per_node` is automatically
        # set to 1 and the value is passed to torchrun's nproc_per_node argument.
        launcher="torchrun",
        container_mounts=["/lustre:/lustre",],
        env_vars=env_vars,
    )
    # slurm_executor.setup_lines = env_vars
    # 3. Run your experiment
    # The `run.run` function wraps the task and executor in an Experiment and launches it.
    run.run(
        partial_func,
        executor=slurm_executor,
        name="multinode_llama3_8b_finetune" if model == "llama3-8b" else "multinode_Qwen3_235B_A22B_finetune"
    )