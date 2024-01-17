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


import os

import torch.multiprocessing as mp
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True
except:
    pass

mp.set_start_method("spawn", force=True)
"""
This is the script to convert peft ckpt to .nemo file.

Example usage:

python examples/nlp/language_modeling/tuning/megatron_gpt_peft_ckpt_to_nemo.py \
    trainer.devices=2 \
    trainer.num_nodes=1 \
    trainer.precision=bf16 \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=1 \
    model.peft.peft_scheme=lora \
    ++model.peft.lora_tuning.target_modules=['all'] \
    model.megatron_amp_O2=False \
	model.restore_from_path=<path_to_gpt_nemo_file> \
    ++model.peft.restore_from_ckpt.checkpoint_dir=<path_to_peft_ckpt_dir> \
    ++model.peft.restore_from_ckpt.checkpoint_name=<ckpt_name> \
    ++output_nemo_path=<path_to_the_output_nemo_file>

"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_peft_eval_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()

    model_cfg = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, return_config=True)
    with open_dict(model_cfg):
        # update the model config of the trained model with params we want to set at inference time.
        model_cfg.precision = cfg.trainer.precision
        for key, val in cfg.model.items():
            if key != 'data':
                model_cfg[key] = val

    model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)

    if cfg.model.peft.restore_from_ckpt:
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
    else:
        raise Exception("No PEFT checkpoint is provided")

    model.freeze()
    logging.info(f"Freezing parameters for PEFT eval:\n{model.summarize()}")
    model.setup_complete = True

    model.save_to(cfg.output_nemo_path)


if __name__ == "__main__":
    main()
