import nemo_run as run

from nemo.collections import vlm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = vlm.mllama_11b.finetune_recipe(
        dir="/checkpoints/mllama",  # Path to store checkpoints
        name="mllama",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme="lora",
    )
    recipe.trainer.max_steps = 100
    recipe.trainer.val_check_interval = 100
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {}

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def run_training():
    recipe = configure_recipe()
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_training()
