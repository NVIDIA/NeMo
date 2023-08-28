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

from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AttnAdapterConfig,
    AttnPtuningAdapterConfig,
    IA3AdapterConfig,
    LoraAdapterConfig,
    PtuningAdapterConfig,
)
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)

"""
This is the script to train an Adapter infused GPT Model for text generation.
A base GPT Model is required as a starting point. This script will then insert
Adapters into each Transformer layer and will train/update only these adapters
during training. The base GPT Model weights will remain frozen.

During training this script will only save the newly trained Adapter weights
in checkpoints. At the end of training a .nemo file of Adapter weights will 
be saved.

Usage:
    Assuming the base model is a 125m GPT Model, with TP=1, PP=1:
    a. run a training run for a base gpt nemo file:
        python megatron_gpt_adapter_tuning.py \
            "model.data.train_ds=[PATH TO TRAINING JSONL FILE]",
            "model.data.validation_ds=[PATH TO VALIDATION JSONL FILE]",
            model.language_model_path="PATH TO BASE GPT MODEL .nemo FILE"
            name="NAME OF TRAINING RUN"
            exp_manager.exp_dir="DIR TO SAVE CHECKPOINTS and .nemo FILE",
            trainer.max_epochs=2
"""

peft_config_map = {
    "adapter": AttnAdapterConfig,
    "ia3": IA3AdapterConfig,
    "ptuning": PtuningAdapterConfig,
    "adapter_and_ptuning": AttnPtuningAdapterConfig,
    "lora": LoraAdapterConfig,
}


def _modify_config(t5_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (t5_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(t5_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(t5_cfg):
        t5_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        t5_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        t5_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        t5_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        t5_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        t5_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        t5_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        t5_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        t5_cfg.data = cfg.model.data
        t5_cfg.optim = cfg.model.optim
        t5_cfg.precision = cfg.trainer.precision
        t5_cfg.answer_only_loss = cfg.model.answer_only_loss
        t5_cfg.restore_from_path = cfg.model.restore_from_path
        t5_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        t5_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        t5_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        t5_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
        t5_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
        t5_cfg.ffn_dropout = cfg.model.ffn_dropout
        t5_cfg.peft = cfg.model.peft

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(t5_cfg)
            t5_cfg.cfg = t5_cfg

    return t5_cfg


def build_config(cfg, trainer):
    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        trainer.ckpt_path = cfg.model.resume_from_checkpoint
    logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    assert cfg.model.restore_from_path, "PEFT training needs a trained base model present."

    base_model_cfg = MegatronT5SFTModel.restore_from(
        restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True,
    )
    base_model_cfg = _modify_config(base_model_cfg, cfg, add_cfg_to_tree=False)
    return base_model_cfg


@hydra_runner(config_path="conf", config_name="megatron_t5_peft_tuning_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    base_model_cfg = build_config(cfg, trainer)
    model = MegatronT5SFTModel.restore_from(
        restore_path=cfg.model.restore_from_path, trainer=trainer, override_config_path=base_model_cfg,
    )

    AdapterConfig = peft_config_map[base_model_cfg.peft.peft_scheme]
    model.add_adapters(AdapterConfig(base_model_cfg))

    trainer.fit(model)


if __name__ == '__main__':
    main()
