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

import torch.multiprocessing as mp
import os
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

mp.set_start_method("spawn", force=True)

def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        gpt_cfg.micro_batch_size = cfg.model.micro_batch_size
        gpt_cfg.rampup_batch_size = cfg.model.get('rampup_batch_size', None)
        gpt_cfg.global_batch_size = cfg.model.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        gpt_cfg.num_micro_batches_with_partial_activation_checkpoints = cfg.model.get("num_micro_batches_with_partial_activation_checkpoints", None)
        gpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get("activations_checkpoint_layers_per_pipeline", None)
        gpt_cfg.use_cpu_initialization = cfg.model.get("use_cpu_initialization", False)
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.seed  = cfg.model.seed
        gpt_cfg.transformer_engine  = cfg.model.get('transformer_engine', False)
        gpt_cfg.use_flash_attention  = cfg.model.get('use_flash_attention', False)
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.grad_allreduce_chunk_size_mb = cfg.model.get('grad_allreduce_chunk_size_mb', 125)
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
        gpt_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
        gpt_cfg.ffn_dropout = cfg.model.get('ffn_dropout', 0.0)
        gpt_cfg.grad_div_ar_fusion =cfg.model.get('grad_div_ar_fusion', False)
        gpt_cfg.gradient_accumulation_fusion = cfg.model.get('gradient_accumulation_fusion', False)
        gpt_cfg.bias_activation_fusion = cfg.model.get('bias_activation_fusion', False)
        gpt_cfg.bias_dropout_add_fusion = cfg.model.get('bias_dropout_add_fusion', False)
        gpt_cfg.masked_softmax_fusion = cfg.model.get('masked_softmax_fusion', False)
        #sft_cls = MegatronLLAMAModel
        #gpt_cfg.target = f"{sft_cls.__module__}.{sft_cls.__name__}"

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg

def load_from_nemo(cls, cfg, trainer, gpt_cfg, modify_confg_fn):
    gpt_cfg = modify_confg_fn(gpt_cfg, cfg, add_cfg_to_tree=False)
    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
    model = cls.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=gpt_cfg,
        save_restore_connector=save_restore_connector,
    )
    return model

@hydra_runner(config_path="conf", config_name="megatron_llama_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    if cfg.model.get('restore_from_path', None):
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.restore_from_path):
            save_restore_connector.model_extracted_dir = cfg.model.restore_from_path
        gpt_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        gpt_cfg = _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False)
        model = load_from_nemo(MegatronGPTModel, cfg, trainer, gpt_cfg, modify_confg_fn=_modify_config)
    else:
        model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
