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

import clip
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import MegatronCLIPModel
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_scorefusion_models import MegatronCLIPScoreFusionModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.data.clip.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRCandidatePoolDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolCollator,
    Mode,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler



def get_tokenizer(tokenizer):
    def tokenizer_wrapper(txt):
        txt_tensor = tokenizer(txt, context_length=77, truncate=True)
        return txt_tensor

    return tokenizer_wrapper

@hydra_runner(config_path="conf", config_name="megatron_clipscorefusion_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
    ) * cfg.model.micro_batch_size == cfg.model.global_batch_size, (
        "Gradient accumulation is not supported in CLIP yet."
    )

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    
    model = MegatronCLIPScoreFusionModel(cfg.model, trainer)
    trainer.fit(model)


if __name__ == '__main__':
    main()
