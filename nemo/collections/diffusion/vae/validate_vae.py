# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import nemo_run as run
from nemo.collections import llm
from nemo.collections.diffusion.vae.train_vae import train_vae


@run.cli.factory(target=llm.validate)
def validate_vae() -> run.Partial:
    recipe = train_vae()
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
    run.cli.main(llm.validate, default_factory=validate_vae)
