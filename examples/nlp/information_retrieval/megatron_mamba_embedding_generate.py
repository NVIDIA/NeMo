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
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.information_retrieval.megatron_mamba_embedding_model import (
    MegatronMambaEmbeddingModel,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import inject_model_parallel_rank
import os

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_mamba_embedder_generate_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()

    model_cfg = MegatronMambaEmbeddingModel.merge_inference_cfg(cfg.model.restore_from_path, cfg)

    # with open_dict(model_cfg):
    #     model_cfg.data.return_output_tensors = True
    #     model_cfg.post_process = False

    # model = MegatronMambaEmbeddingModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    logging.info(f"Loading model from {cfg.model.restore_from_path}")
    model = MegatronMambaEmbeddingModel.restore_from(
        restore_path=cfg.model.restore_from_path, trainer=trainer, override_config_path=model_cfg, strict=False
    )

    if cfg.model.peft.restore_from_path:
        model.load_adapters(cfg.model.peft.restore_from_path)
    elif cfg.model.peft.restore_from_ckpt.checkpoint_dir and cfg.model.peft.restore_from_ckpt.checkpoint_name:
        peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]
        checkpoint_path = os.path.join(
            cfg.model.peft.restore_from_ckpt.checkpoint_dir, cfg.model.peft.restore_from_ckpt.checkpoint_name
        )
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(
                os.path.join(
                    cfg.model.peft.restore_from_ckpt.checkpoint_dir, cfg.model.peft.restore_from_ckpt.checkpoint_name
                )
            )
            model.load_adapters(checkpoint_path, peft_cfgs=peft_cfg_cls(model_cfg))
        else:
            raise NotImplementedError("distributed checkpointing of PEFT weights is not supported")
        
    if not cfg.model.get('use_flash_attention', False):
        cfg.inference.compute_attention_mask = True
    config = OmegaConf.to_container(cfg.inference, resolve=True)
    model.set_inference_config(config)

    # if not cfg.server:
    trainer.test(model)
    # else:
    # use_inference_server(cfg, model, trainer)


if __name__ == "__main__":
    main()
