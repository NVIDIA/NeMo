import nemo_run as run

from nemo.collections import llm, vlm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 8):
    recipe = vlm.llava15_7b.finetune_recipe(
        dir="/checkpoints/llava",  # Path to store checkpoints
        name="llava_ft",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        peft_scheme="none",
    )
    recipe.trainer.max_steps = 100
    recipe.trainer.val_check_interval = 100
    recipe.model.config.freeze_vision_model = True
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 8) -> run.LocalExecutor:
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
