# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import tempfile
from pathlib import Path

import pytest
import os
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from megatron.core import parallel_state
import pytorch_lightning as pl
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.multimodal.models.speechllm_models import (
    ModularizedSpeechGPTModel,
)
from nemo.collections.asr.models.hybrid_asr_tts_models import ASRWithTTSModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)


def setup_module():
    # init model parallel needed for LLM loss
    init_method = 'tcp://'
    master_ip = 'localhost'
    master_port = '6000'
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend='gloo', world_size=1, rank=0, init_method=init_method)
    parallel_state.initialize_model_parallel(1, 1)

@pytest.fixture
def llm_model_config():
    this_test_dir = os.path.dirname(os.path.abspath(__file__))
    # Although most of the stuff in model is loaded from ckpt, we need configs
    # for e.g. cfg.model.optim
    config = OmegaConf.load(
        os.path.join(
            this_test_dir,
            "../../../examples/multimodel/conf/speechllm/modularized_speech_gpt_config.yaml",
        )
    )
    # TODO(zhehuai): update train_ds and validation_ds
    # wget  -nc --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_gpt_345m/versions/1/files/megatron_gpt_345m.nemo -O /home/TestData/nlp/megatron_gpt/megatron_gpt_345m.nemo
    config.model.language_model_path = (
        "/home/TestData/nlp/megatron_gpt/megatron_gpt_345m.nemo"
    )
    return config


@pytest.fixture
def trainer_config():
    config_trainer = DictConfig({})
    
    if torch.cuda.is_available():
        accelerator = "gpu" 
        torch.set_default_device('cuda')
    else:
        accelerator = "cpu"
    config_trainer.accelerator = accelerator
    config_trainer.devices = 1
    config_trainer.num_nodes = 1
    config_trainer.max_epochs = 4
    config_trainer.val_check_interval = 1.0

    # for PyTorch Native AMP set precision=16
    config_trainer.precision = 16 if torch.cuda.is_available() else 32

    # setup cluster environment parameters"
    # use torch elastic cluster environment so `create_process_externally` is True
    # the launcher is set to None. It will not try to spawn new processes.
    # It won't create the misconfiguration error because of the `interactive session`
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    strategy = NLPDDPStrategy(
    )
    plugins = [TorchElasticEnvironment()]
    trainer = pl.Trainer(logger=None, plugins=plugins, strategy=strategy, **config_trainer)
    return trainer, config_trainer


@pytest.fixture
def perception_model_config():
    preprocessor = {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"
    }
    encoder = {
        "_target_": "nemo.collections.asr.modules.ConformerEncoder",
        "feat_in": 64,
        "n_layers": 8,
        "d_model": 64,
        "self_attention_model": "rel_pos_local_attn",
        "att_context_size": [128, 128],
    }

    model_config = DictConfig(
        {
            "_target_": "nemo.collections.multimodal.modules.speechllm_perception.AudioPerceptionModel",
            "preprocessor": DictConfig(preprocessor),
            "encoder": DictConfig(encoder),
            "matcher": DictConfig(encoder),
            "d_model": 1024,
        }
    )
    return model_config

class TestModularizedSpeechGPTModel:
    @pytest.mark.unit
    def test_init_and_train(
        self, llm_model_config, perception_model_config, trainer_config
    ):
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedSpeechGPTModel(cfg=llm_model_config.model, trainer=trainer)

        assert isinstance(model.frozen_model, MegatronGPTModel)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model.nemo")
            model.train()
            model.save_to(save_path)

    @pytest.mark.unit
    def test_training_step(
        self, llm_model_config, perception_model_config, trainer_config
    ):
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedSpeechGPTModel(cfg=llm_model_config.model, trainer=trainer)
        model.cuda()

        model.train()
        pl.seed_everything(1)
        signal = torch.randn(2, 64000)
        signal_len = torch.from_numpy(np.array([64000, 64000]))
        transcript = torch.randn(2, 4).int()
        transcript_length = torch.from_numpy(np.array([3, 2]))
        batch = signal, signal_len, transcript, transcript_length
        loss_mean = model.training_step(batch, None)
        assert np.allclose(loss_mean.cpu().detach().numpy(), 5.1678314)

    @pytest.mark.unit
    def test_validation_step(
        self, llm_model_config, perception_model_config, trainer_config
    ):
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedSpeechGPTModel(cfg=llm_model_config.model, trainer=trainer)
        model.cuda()

        model.train()
        pl.seed_everything(1)
        signal = torch.randn(2, 64000)
        signal_len = torch.from_numpy(np.array([64000, 64000]))
        transcript = torch.randn(2, 4).int()
        transcript_length = torch.from_numpy(np.array([3, 2]))
        batch = signal, signal_len, transcript, transcript_length
        loss_mean = model.validation_step(batch, None)
        assert np.allclose(loss_mean['loss'].cpu().detach().numpy(), 5.1694)

    # TODO(zhehuai): test ckpt restore
