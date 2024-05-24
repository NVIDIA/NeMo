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


# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import thunder
from thunder.examine import examine

torch._dynamo.config.suppress_errors = True

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model = MegatronGPTModel(cfg.model, trainer)

    # input_ids = torch.ones((4, 128), dtype=torch.int64, device='cuda')
    # position_ids = torch.ones((4, 128), dtype=torch.int64, device='cuda')
    # attention_mask = None
    # labels = torch.zeros((4, 128), dtype=torch.int64, device='cuda')
    # caal = None
    # model.model.to('cuda:0')
    # examine(model.model, input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, 
    #         labels=labels, checkpoint_activations_all_layers=caal)
    # import sys
    # sys.exit(1)

    trainer.fit(model)


if __name__ == '__main__':
    main()
