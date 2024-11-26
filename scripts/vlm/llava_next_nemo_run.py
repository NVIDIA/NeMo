import nemo_run as run

from nemo.collections import llm, vlm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 8, pretrain=False):
    if pretrain:
        recipe = vlm.llava_next_7b.pretrain_recipe(
            dir="./outputs/checkpoints/llava",  # Path to store checkpoints
            name="llava_pretrain",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
        )
    else:
        recipe = vlm.llava_next_7b.finetune_recipe(
            dir="./outputs/checkpoints/llava",  # Path to store checkpoints
            name="llava_finetune",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
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


def run_pretraining():
    recipe = configure_recipe(pretrain=True)
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


def run_finetuning():
    recipe = configure_recipe(pretrain=False)
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()
    # run_finetuning()
