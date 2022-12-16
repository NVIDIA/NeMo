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

from pathlib import Path

import pytest
import torch
from einops import rearrange
from omegaconf import DictConfig

from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel
from nemo.collections.tts.modules.spectrogram_enhancer import mask


@pytest.fixture
def enhancer_config():
    n_bands = 80
    latent_dim = 192
    style_depth = 4
    network_capacity = 16
    fmap_max = 192

    config = {
        "model": {
            "n_bands": n_bands,
            "latent_dim": latent_dim,
            "style_depth": style_depth,
            "network_capacity": network_capacity,
            "mixed_prob": 0.9,
            "fmap_max": fmap_max,
            "generator": {
                "_target_": "nemo.collections.tts.modules.spectrogram_enhancer.Generator",
                "n_bands": n_bands,
                "latent_dim": latent_dim,
                "network_capacity": network_capacity,
                "style_depth": style_depth,
                "fmap_max": fmap_max,
            },
            "discriminator": {
                "_target_": "nemo.collections.tts.modules.spectrogram_enhancer.Discriminator",
                "n_bands": n_bands,
                "network_capacity": network_capacity,
                "fmap_max": fmap_max,
            },
            "spectrogram_min_value": -13.18,
            "spectrogram_max_value": 4.78,
            "consistency_loss_weight": 10.0,
            "gradient_penalty_loss_weight": 10.0,
            "gradient_penalty_loss_every_n_steps": 4,
            "spectrogram_predictor_path": None,
        },
        "generator_opt": {"_target_": "torch.optim.Adam", "lr": 2e-4, "betas": [0.5, 0.9]},
        "discriminator_opt": {"_target_": "torch.optim.Adam", "lr": 2e-4, "betas": [0.5, 0.9]},
    }

    return DictConfig(config)


@pytest.fixture
def enhancer_config_with_fastpitch(fastpitch_model_path, test_data_dir):
    test_data_dir = Path(test_data_dir)

    n_bands = 80
    latent_dim = 192
    style_depth = 4
    network_capacity = 16
    fmap_max = 192

    config = {
        "model": {
            "n_bands": n_bands,
            "latent_dim": latent_dim,
            "style_depth": style_depth,
            "network_capacity": network_capacity,
            "mixed_prob": 0.9,
            "fmap_max": fmap_max,
            "generator": {
                "_target_": "nemo.collections.tts.modules.spectrogram_enhancer.Generator",
                "n_bands": n_bands,
                "latent_dim": latent_dim,
                "network_capacity": network_capacity,
                "style_depth": style_depth,
                "fmap_max": fmap_max,
            },
            "discriminator": {
                "_target_": "nemo.collections.tts.modules.spectrogram_enhancer.Discriminator",
                "n_bands": n_bands,
                "network_capacity": network_capacity,
                "fmap_max": fmap_max,
            },
            "spectrogram_min_value": -13.18,
            "spectrogram_max_value": 4.78,
            "consistency_loss_weight": 10.0,
            "gradient_penalty_loss_weight": 10.0,
            "gradient_penalty_loss_every_n_steps": 4,
            "spectrogram_model_path": fastpitch_model_path,
            "train_ds": {
                "dataset": {
                    "manifest_filepath": str(test_data_dir / "tts/mini_ljspeech/manifest.json"),
                    "sup_data_path": str(test_data_dir / "tts/mini_ljspeech/sup"),
                },
                "dataloader_params": {"batch_size": 3},
            },
        },
        "generator_opt": {"_target_": "torch.optim.Adam", "lr": 2e-4, "betas": [0.5, 0.9]},
        "discriminator_opt": {"_target_": "torch.optim.Adam", "lr": 2e-4, "betas": [0.5, 0.9]},
    }

    return DictConfig(config)


@pytest.fixture
def fastpitch_model():
    model = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
    return model


@pytest.fixture
def fastpitch_model_path(fastpitch_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("spectrogram-enhancer") / "fastpitch-for-tests.nemo"
    fastpitch_model.save_to(path)
    return path


@pytest.fixture
def enhancer(enhancer_config):
    return SpectrogramEnhancerModel(cfg=enhancer_config.model)


@pytest.fixture
def enhancer_with_fastpitch(enhancer_config_with_fastpitch):
    return SpectrogramEnhancerModel(cfg=enhancer_config_with_fastpitch.model)


@pytest.fixture
def sample_input(batch_size=15, max_length=1000):
    generator = torch.Generator()
    generator.manual_seed(0)

    lengths = torch.randint(max_length // 4, max_length - 7, (batch_size,), generator=generator)
    condition = torch.randn((batch_size, 80, 1000), generator=generator)

    condition = mask(condition, lengths)

    return condition, lengths


@pytest.mark.unit
def test_pad_spectrogram(enhancer, sample_input):
    condition, lengths = sample_input
    output = enhancer.pad_spectrogram(condition)

    assert output.size(-1) >= condition.size(-1)


@pytest.mark.unit
def test_spectrogram_norm_unnorm(enhancer: SpectrogramEnhancerModel, sample_input):
    condition, lengths = sample_input
    same_condition = enhancer.unnormalize_spectrograms(enhancer.normalize_spectrograms(condition, lengths), lengths)
    assert torch.allclose(condition, same_condition, atol=1e-5)


@pytest.mark.unit
def test_spectrogram_unnorm_norm(enhancer: SpectrogramEnhancerModel, sample_input):
    condition, lengths = sample_input
    same_condition = enhancer.normalize_spectrograms(enhancer.unnormalize_spectrograms(condition, lengths), lengths)
    assert torch.allclose(condition, same_condition, atol=1e-5)


@pytest.mark.unit
def test_spectrogram_norm_unnorm_dont_look_at_padding(enhancer: SpectrogramEnhancerModel, sample_input):
    condition, lengths = sample_input
    same_condition = enhancer.unnormalize_spectrograms(enhancer.normalize_spectrograms(condition, lengths), lengths)
    for i, length in enumerate(lengths.tolist()):
        assert torch.allclose(condition[i, :, :length], same_condition[i, :, :length], atol=1e-5)


@pytest.mark.unit
def test_spectrogram_unnorm_norm_dont_look_at_padding(enhancer: SpectrogramEnhancerModel, sample_input):
    condition, lengths = sample_input
    same_condition = enhancer.normalize_spectrograms(enhancer.unnormalize_spectrograms(condition, lengths), lengths)
    for i, length in enumerate(lengths.tolist()):
        assert torch.allclose(condition[i, :, :length], same_condition[i, :, :length], atol=1e-5)


@pytest.mark.unit
def test_generator_pass_keeps_size(enhancer: SpectrogramEnhancerModel, sample_input):
    condition, lengths = sample_input
    output = enhancer.forward(condition=condition, lengths=lengths)

    assert output.shape == condition.shape


@pytest.mark.unit
def test_discriminator_pass(enhancer: SpectrogramEnhancerModel, sample_input):
    condition, lengths = sample_input
    condition = rearrange(condition, "b c l -> b 1 c l")
    logits = enhancer.discriminator(x=condition, condition=condition, lengths=lengths)

    assert logits.shape == lengths.shape


@pytest.mark.unit
def test_nemo_save_load(enhancer: SpectrogramEnhancerModel, tmp_path):
    path = tmp_path / "test-enhancer-save-load.nemo"

    enhancer.save_to(path)
    return SpectrogramEnhancerModel.restore_from(path)


@pytest.mark.with_downloads
@pytest.mark.unit
def test_nemo_save_load_enhancer_with_fastpitch(enhancer_with_fastpitch: SpectrogramEnhancerModel, tmp_path):
    path = tmp_path / "test-enhancer-save-load.nemo"

    enhancer_with_fastpitch.save_to(path)
    restored_enhancer = SpectrogramEnhancerModel.restore_from(path)
    assert restored_enhancer.spectrogram_model is None


# @pytest.mark.with_downloads
# @pytest.mark.unit
def fastpitch_loads_data_inside_enhancer(enhancer_with_fastpitch: SpectrogramEnhancerModel):
    assert len(enhancer_with_fastpitch._train_dl.dataset) > 0


# @pytest.mark.with_downloads
# @pytest.mark.unit
def enhancer_with_fastpitch_training_step_discriminator(enhancer_with_fastpitch: SpectrogramEnhancerModel):
    batch = next(iter(enhancer_with_fastpitch._train_dl))
    enhancer_with_fastpitch.training_step(batch, batch_idx=0, optimizer_idx=0)


# @pytest.mark.with_downloads
# @pytest.mark.unit
def enhancer_with_fastpitch_training_step_generator(enhancer_with_fastpitch: SpectrogramEnhancerModel):
    batch = next(iter(enhancer_with_fastpitch._train_dl))
    enhancer_with_fastpitch.training_step(batch, batch_idx=0, optimizer_idx=1)


@pytest.mark.with_downloads
@pytest.mark.unit
def test_enhancer_with_fastpitch(enhancer_with_fastpitch: SpectrogramEnhancerModel):
    # long setup for enhancer_with_fastpitch, around 1 min
    fastpitch_loads_data_inside_enhancer(enhancer_with_fastpitch)
    enhancer_with_fastpitch_training_step_discriminator(enhancer_with_fastpitch)
    enhancer_with_fastpitch_training_step_generator(enhancer_with_fastpitch)
