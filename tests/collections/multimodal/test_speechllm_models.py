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

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from megatron.core import parallel_state
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.asr.models.hybrid_asr_tts_models import ASRWithTTSModel
from nemo.collections.multimodal.models import speechllm_models
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


class ModularizedAudioGPTModel(speechllm_models.ModularizedAudioGPTModel):
    # disable logging to avoid MisconfigurationException
    def log(self, *args, **kwargs):
        pass


def setup_module():
    pl.seed_everything(1)
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
        os.path.join(this_test_dir, "../../../examples/multimodel/conf/speechllm/modularized_speech_gpt_config.yaml",)
    )
    # TODO(zhehuai): move the following to Test /home/TestData
    config.model.restore_from_path = "/root/home/works/TestData/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo"
    config.model.micro_batch_size = 2
    config.model.global_batch_size = 2
    config.model.data.validation_ds.manifest_filepath = (
        '/root/home/works/TestData/datasets/LibriSpeech/dev_clean_cleaned.json'
    )
    config.model.data.train_ds.manifest_filepath = (
        '/root/home/works/TestData/datasets/LibriSpeech/dev_clean_cleaned.json'
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
    config_trainer.max_steps = 1
    config_trainer.val_check_interval = 1.0

    # for PyTorch Native AMP set precision=16
    config_trainer.precision = 32

    # setup cluster environment parameters"
    # use torch elastic cluster environment so `create_process_externally` is True
    # the launcher is set to None. It will not try to spawn new processes.
    # It won't create the misconfiguration error because of the `interactive session`
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    strategy = NLPDDPStrategy()
    plugins = [TorchElasticEnvironment()]
    trainer = pl.Trainer(logger=False, plugins=plugins, strategy=strategy, **config_trainer)
    return trainer, config_trainer


@pytest.fixture
def perception_model_config():
    preprocessor = {"_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"}
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
            "modality_adapter": DictConfig(encoder),
            "output_dim": 1024,
        }
    )
    return model_config


@pytest.fixture
def test_batch():
    signal_len = torch.from_numpy(np.array([64000, 64000]))
    transcript = torch.arange(10).reshape(2, 5).int()
    tokens = transcript[:, :-1]
    labels = transcript[:, 1:]
    transcript_length = torch.Tensor([3, 2]).int()
    # assuming context_lengths = [1, 1]
    loss_mask = torch.Tensor([[0, 1, 1, 0], [0, 1, 0, 0]])
    batch = {
        'audio_signal_length': signal_len,
        'tokens': tokens,
        'tokens_length': transcript_length,
        'contexts': torch.arange(260).reshape(2, 130).int(),
        'context_lengths': torch.Tensor([1, 1]).int(),
        'labels': labels,
        'answers': labels,
        'loss_mask': loss_mask,
    }
    batch['audio_signal'] = torch.randn([2, 64000])
    return batch


class TestModularizedAudioGPTModel:
    @pytest.mark.unit
    def test_init_and_train(self, llm_model_config, perception_model_config, trainer_config):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)

        assert isinstance(model.model, GPTModel)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model.nemo")
            model.train()
            model.save_to(save_path)

    @pytest.mark.unit
    def test_prepare_llm_input(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.train()
        batch = {key: val.cuda(non_blocking=True) for key, val in test_batch.items()}
        encoder_input, attention_mask, labels, loss_mask, encoder_length = model.prepare_llm_input(batch)
        assert encoder_input.shape == (17, 2, 768)
        assert np.allclose(encoder_input.sum().cpu().detach().numpy(), 15.783691)
        assert attention_mask.shape == (2, 1, 17, 17)
        assert labels.shape == (2, 17)
        assert np.allclose(loss_mask.sum(axis=1).cpu().numpy(), [2, 1])
        assert np.allclose(encoder_length.cpu().numpy(), (16, 15))

    @pytest.mark.unit
    def test_training_step(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.on_train_start()
        model.setup()
        model.train()
        loss_mean = model.training_step(iter([test_batch]), None)
        assert np.allclose(loss_mean.cpu().detach().numpy(), 5.7052)

    @pytest.mark.unit
    def test_validation_step(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.train()
        batch = {key: val.cuda(non_blocking=True) for key, val in test_batch.items()}
        loss_mean = model.validation_step(iter([batch]), 0)
        assert np.allclose(loss_mean['loss'].cpu().detach().numpy(), 5.7052)

    @pytest.mark.unit
    def test_predict_step(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularizedAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.train()
        batch = {key: val.cuda(non_blocking=True) for key, val in test_batch.items()}
        response = model.predict_step(batch, 0, 0)
        assert np.allclose(np.mean(response['token_ids']), 2429.3169014084506)
