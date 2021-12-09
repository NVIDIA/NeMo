# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.config_utils import update_model_config

@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    default_cfg = OmegaConf.to_yaml(cfg)
    cfg = update_model_config(default_cfg, cfg)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = [NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)]

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)
    model = MegatronGPTModel.restore_from(cfg.restore_from_path, cfg.model, trainer=trainer)

    # Init all new prompts
    for new_prompt in cfg.model.new_prompts:
        if new_prompt.init_method == 'random':
            model.init_prompt_from_random(new_prompt.tag)
        else:
            model.init_prompt_from_text(new_prompt.tag, new_prompt.init_text)

    logging.info(f'\nCurrent soft prompts include {model.get_prompt_table()}')
    trainer.fit(model)

if __name__ == '__main__':
    main()
