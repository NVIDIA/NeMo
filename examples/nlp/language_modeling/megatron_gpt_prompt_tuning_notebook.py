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
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, NLPDDPPlugin, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from nemo.utils.config_utils import update_model_config

# +
cfg = OmegaConf.load("conf/megatron_gpt_config.yaml")

# Set current model params
cfg.model.encoder_seq_length = 2048

# Set prompt tuning params
cfg.model.optim.lr = 2e-4
cfg.model.optim.sched.min_lr = 2e-6
cfg.model.use_soft_prompts = True
cfg.model.prompt_length = 10
cfg.model.data.train_ds = 'prompt_tuning_ner_train.json'
cfg.model.data.valid_ds = 'prompt_tuning_ner_val.json'
cfg.model.data.test_ds = 'prompt_tuning_ner_test.json'
cfg.model.data.batch_size = 32
cfg.model.data.data_prefix = None
cfg.model.optim.sched.warmup_steps = 50
cfg.model.optim.sched.constant_steps = 100
cfg.trainer.max_steps = 200
cfg.restore_from_path = 'megatron_gpt.nemo'

new_config = open('conf/gpt_prompt_tuning_config.yaml', 'w')
OmegaConf.save(config=cfg, f=new_config)

# +
plugins = [NLPDDPPlugin(num_nodes=cfg.trainer.num_nodes)]

# if cfg.trainer.precision == 16:
#     scaler = GradScaler(
#         init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
#         growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
#     )
#     plugins.append(NativeMixedPrecisionPlugin(precision=16, device='cuda', scaler=scaler))

if cfg.get('cluster_type', None) == 'BCP':
    plugins.append(TorchElasticEnvironment())

trainer = Trainer(plugins=plugins, **cfg.trainer)

exp_manager(trainer, cfg.exp_manager)

#model = MegatronGPTModel(cfg.model, trainer)
model = MegatronGPTModel.restore_from(cfg.restore_from_path, cfg.model, trainer=trainer)
# -

model.get_prompt_table()

#model.init_prompt_from_random("NER-Yes-No", torch.nn.init.normal_)
model.init_prompt_from_text("NER-Yes-No", "find entity name")

model.get_prompt_table()

model.init_prompt_from_text("NER-Complete", "find entity name")

model.get_prompt_table()

model.prompt_tuning_freeze()

trainer.fit(model)


