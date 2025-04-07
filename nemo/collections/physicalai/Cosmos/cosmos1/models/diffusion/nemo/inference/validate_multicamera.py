# The early-access software is governed by the NVIDIA Evaluation License Agreement – EA Cosmos Code (v. Feb 2025).
# The license reference will be the finalized version of the license linked above.

import os

import nemo_run as run
from huggingface_hub import snapshot_download
from nemo.collections import llm

from cosmos1.models.diffusion.nemo.post_training.multicamera import cosmos_multicamera_diffusion_7b_text2world_finetune


@run.cli.factory(target=llm.validate)
def cosmos_multicamera_diffusion_7b_text2world_validate() -> run.Partial:
    recipe = cosmos_multicamera_diffusion_7b_text2world_finetune()

    # Checkpoint load
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Video2World", allow_patterns=["nemo/*"]), "nemo"
    )  # path to diffusion model checkpoint

    return run.Partial(
        llm.validate,
        model=recipe.model,
        data=recipe.data,
        trainer=recipe.trainer,
        log=recipe.log,
        optim=recipe.optim,
        tokenizer=None,
        resume=recipe.resume,
        model_transform=None,
    )


if __name__ == "__main__":
    run.cli.main(llm.validate, default_factory=cosmos_multicamera_diffusion_7b_text2world_validate)
