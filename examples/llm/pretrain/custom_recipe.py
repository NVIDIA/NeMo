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
from nemo.collections.llm.recipes import llama3_8b, llama3_70b, hf_auto_model_for_causal_lm

from nemo.collections.llm.recipes.log.default import default_log
from data_utils import squad
from lightning.pytorch.loggers import WandbLogger


def custom_llama3_8b():
    pretrain = llama3_8b.pretrain_recipe(num_nodes=1, num_gpus_per_node=8)

    pretrain.trainer.val_check_interval = 400
    pretrain.log.ckpt.save_top_k = -1
    pretrain.log.ckpt.every_n_train_steps = 400

    pretrain.trainer.max_steps = 1000

    return pretrain


def custom_llama3_70b():
    pretrain = llama3_70b.pretrain_recipe(num_nodes=1, num_gpus_per_node=8)

    pretrain.trainer.val_check_interval = 400
    pretrain.log.ckpt.save_top_k = -1
    pretrain.log.ckpt.every_n_train_steps = 400

    pretrain.trainer.max_steps = 1000

    return pretrain

def custom_hf_auto_model_for_causal_lm(wandb_project_name = None):
    model_name = "meta-llama/Llama-3.2-1B"
    num_gpus_per_node = 2
    pretrain = hf_auto_model_for_causal_lm.pretrain_recipe(model_name = model_name,num_nodes=1, num_gpus_per_node=2)

    pretrain.trainer.max_steps = 1000
    pretrain.trainer.val_check_interval = 400
    pretrain.log.ckpt.save_top_k = -1

    # Add a custom dataset
    data=squad(llm.HFAutoModelForCausalLM.configure_tokenizer(model_name=model_name), gbs=num_gpus_per_node)
    pretrain.data = data

    # Add a custom logger
    wandb_project_name = "nemo2"
    if wandb_project_name is not None:
        model = '_'.join(model_name.split('/')[-2:])
        try:
            wandb_logger = WandbLogger(
                project=wandb_project_name,
                name=f'{model}_dev{num_gpus_per_node}_strat_ddp',
            )
            pretrain.log = default_log(wandb_logger=wandb_logger)
        except Exception as e:
            print(f"Warning: WandbLogger initialization failed with error: {e}")
    return pretrain

if __name__ == "__main__":
    # When running this file, it will run the `custom_llama3_8b` recipe

    # To select the `custom_llama3_70b` recipe, use the following command:
    #   python custom_recipe.py --factory custom_llama3_70b
    #   This will automatically call the custom_llama3_70b that's defined above

    # Note that any parameter can be overwritten by using the following syntax:
    # python custom_recipe.py trainer.max_steps=2000

    # You can even apply transformations when triggering the CLI as if it's python code
    # python custom_recipe.py "trainer.max_steps*=2"

    run.cli.main(llm.pretrain, default_factory=custom_llama3_8b)
