# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_t5_adapter_model import (
    MegatronT5AdapterLearningModel,
    MegatronT5LoraModel,
    MegatronT5InfusedAdapterModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PEFTSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.model_utils import inject_model_parallel_rank

mp.set_start_method("spawn", force=True)

"""

"""


def _modify_config(t5_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original t5 pre-training config (t5_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(t5_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(t5_cfg):
        t5_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        t5_cfg.micro_batch_size = cfg.model.micro_batch_size
        t5_cfg.global_batch_size = cfg.model.global_batch_size
        t5_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        t5_cfg.activations_checkpoint_granularity = cfg.model.get("activations_checkpoint_granularity", None)
        t5_cfg.activations_checkpoint_num_layers = cfg.model.get("activations_checkpoint_num_layers", None)
        t5_cfg.activations_checkpoint_method = cfg.model.get("activations_checkpoint_method", None)
        t5_cfg.data = cfg.model.data
        t5_cfg.optim = cfg.model.optim
        t5_cfg.precision = cfg.trainer.precision
        t5_cfg.answer_only_loss = cfg.model.answer_only_loss
        t5_cfg.language_model_path = cfg.model.language_model_path
        t5_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        t5_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        t5_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        t5_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.0)
        t5_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.0)
        t5_cfg.ffn_dropout = cfg.model.ffn_dropout
        t5_cfg.peft = cfg.model.peft
        peft_cls = _get_peft_scheme(cfg.model)
        t5_cfg.target = f"{peft_cls.__module__}.{peft_cls.__name__}"
        t5_cfg.virtual_prompt_style = cfg.model.virtual_prompt_style
        t5_cfg.task_templates = cfg.model.task_templates
        t5_cfg.new_tasks = cfg.model.new_tasks
        t5_cfg.existing_tasks = cfg.model.existing_tasks
        t5_cfg.nemo_path = cfg.model.nemo_path

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(t5_cfg)
            t5_cfg.cfg = t5_cfg

    return t5_cfg


def _get_peft_scheme(cfg):
    if cfg.peft.peft_scheme == "adapter":
        peft_cls = MegatronT5AdapterLearningModel
    elif cfg.peft.peft_scheme == "ia3":
        peft_cls = MegatronT5InfusedAdapterModel
    elif cfg.peft.peft_scheme == "ptuning":
        peft_cls = MegatronT5PromptLearningModel
    elif cfg.peft.peft_scheme == "lora":
        peft_cls = MegatronT5LoraModel
    else:
        raise RuntimeError("Invalid Peft scheme")
    return peft_cls


def load_from_checkpoint_dir(cls, cfg, trainer, modify_confg_fn):
    app_state = AppState()
    if cfg.model.tensor_model_parallel_size > 1 or cfg.model.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
        app_state.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
        app_state.pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
            app_state.virtual_pipeline_model_parallel_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.model.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.model.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=cfg.model.pipeline_model_parallel_split_rank,
        )
    checkpoint_path = inject_model_parallel_rank(
        os.path.join(cfg.model.pretrained_checkpoint.checkpoint_dir, cfg.model.pretrained_checkpoint.checkpoint_name)
    )
    hparams_file = OmegaConf.load(cfg.model.pretrained_checkpoint.hparams_file)
    t5_cfg = modify_confg_fn(hparams_file.cfg, cfg, add_cfg_to_tree=True)
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
        OmegaConf.save(config=t5_cfg, f=f.name)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint_path, trainer=trainer, hparams_file=f.name,)
        return model


def validate_checkpoint_loading_args(cfg):
    if cfg.checkpoint_dir is None or not os.path.isdir(cfg.checkpoint_dir):
        raise ValueError(f'Checkpoint directory {cfg.checkpoint_dir} does not exist or is not a directory.')
    if cfg.checkpoint_name is None:
        raise ValueError(f'Checkpoint name {cfg.checkpoint_name} is not valid.')
    if cfg.hparams_file is None or not os.path.isfile(cfg.hparams_file):
        raise ValueError(f'Hparams file {cfg.hparams_file} does not exist or is not a file.')


@hydra_runner(config_path="conf", config_name="megatron_t5_peft_tuning_config")
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
                enabled=False
                if cfg.model.pipeline_model_parallel_size > 1
                else True,  # turn off the grad scale for pipeline parallel LM model
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

    if cfg.model.language_model_path:
        base_model_save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.model.language_model_path):
            base_model_save_restore_connector.model_extracted_dir = cfg.model.language_model_path
        base_model_cfg = MegatronT5Model.restore_from(
            restore_path=cfg.model.language_model_path,
            trainer=trainer,
            return_config=True,
            save_restore_connector=base_model_save_restore_connector,
        )
        base_model_cfg = _modify_config(base_model_cfg, cfg, add_cfg_to_tree=False)
        save_restore_connector = PEFTSaveRestoreConnector(
            peft_model_nemo_path=cfg.model.peft.restore_from_path, peft_model_ckpt_path=resume_from_checkpoint
        )
        if os.path.isdir(cfg.model.language_model_path):
            save_restore_connector.model_extracted_dir = cfg.model.language_model_path
        peft_cls = _get_peft_scheme(cfg.model)
        model = peft_cls.restore_from(
            restore_path=cfg.model.language_model_path,
            trainer=trainer,
            override_config_path=base_model_cfg,
            save_restore_connector=save_restore_connector,
        )
    else:
        raise RuntimeError("PEFT training needs a trained base model present.")

    trainer.fit(model)


if __name__ == '__main__':
    main()
