# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import torch
import thunder
from thunder.examine import examine

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="neva_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronNevaModel(cfg.model, trainer)

    # input_ids = torch.ones((2, 340), dtype=torch.int64, device='cuda')
    # position_ids = torch.ones((2, 340), dtype=torch.int64, device='cuda')
    # attention_mask = None
    # labels = torch.zeros((2, 340), dtype=torch.int64, device='cuda')
    # media = torch.randn((2, 1, 1, 3, 224, 224), dtype=torch.float32, device='cuda')
    # caal = None
    # model.model.to('cuda:0')
    # examine(model.model, input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, 
    #         labels=labels, media=media, checkpoint_activations_all_layers=caal)
    # import sys
    # sys.exit(1)

    trainer.fit(model)


if __name__ == '__main__':
    main()
