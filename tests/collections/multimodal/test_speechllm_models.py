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
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
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
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    config_trainer.accelerator = accelerator
    config_trainer.devices = 1
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
        find_unused_parameters=False, no_ddp_communication_hook=True
    )
    plugins = [TorchElasticEnvironment()]
    trainer = pl.Trainer(plugins=plugins, strategy=strategy, **config_trainer)
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
        "d_model": 4,
        "self_attention_model": "rel_pos_local_attn",
        "att_context_size": [128, 128],
        "global_tokens": [0, 1, 4],
        "global_tokens_spacing": [1, 4],
    }

    model_config = DictConfig(
        {
            "_target_": "nemo.collections.multimodal.modules.speechllm_perception.AudioPerceptionModel",
            "preprocessor": DictConfig(preprocessor),
            "encoder": DictConfig(encoder),
            "matcher": DictConfig(encoder),
            "d_model": 512,
        }
    )
    return model_config


class TestASRWithTTSModel:
    @pytest.mark.unit
    def test_init_and_train(
        self, llm_model_config, perception_model_config, trainer_config
    ):
        # TODO(zhehuai): update trainer
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedSpeechGPTModel(cfg=llm_model_config.model, trainer=trainer)

        assert isinstance(model.frozen_model, MegatronGPTModel)
        # TODO(zhehuai): check ASR class
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model.nemo")
            model.train()
            model.save_to(save_path)

    # TODO(zhehuai): test ckpt restore
