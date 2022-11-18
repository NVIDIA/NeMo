from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel


@pytest.fixture
def enhancer_config():
    config = {
        "model": {
            "lr": 2e-4,
            "n_bands": 80,
            "latent_dim": 192,
            "style_depth": 4,
            "network_capacity": 16,
            "mixed_prob": 0.9,
            "fmap_max": 192,
            "spectrogram_min_value": -13.18,
            "spectrogram_max_value": 4.78,
            "spectrogram_predictor_path": None,
        }
    }

    return DictConfig(config)


@pytest.fixture
def enhancer_config_with_fastpitch(fastpitch_model_path, test_data_dir):
    test_data_dir = Path(test_data_dir)
    config = {
        "spectrogram_predictor_path": fastpitch_model_path,
        "model": {
            "lr": 2e-4,
            "n_bands": 80,
            "latent_dim": 192,
            "style_depth": 4,
            "network_capacity": 16,
            "mixed_prob": 0.9,
            "fmap_max": 192,
            "spectrogram_min_value": -13.18,
            "spectrogram_max_value": 4.78,
            "train_ds": {
                "dataset": {
                    "manifest_filepath": str(test_data_dir / "tts/mini_ljspeech/manifest.json"),
                    "sup_data_path": str(test_data_dir / "tts/mini_ljspeech/sup"),
                }
            },
            "validation_ds": {
                "dataset": {
                    "manifest_filepath": str(test_data_dir / "tts/mini_ljspeech/manifest.json"),
                    "sup_data_path": str(test_data_dir / "tts/mini_ljspeech/sup"),
                }
            },
        },
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
    fastpitch = FastPitchModel.restore_from(enhancer_config_with_fastpitch.spectrogram_predictor_path)
    return SpectrogramEnhancerModel(cfg=enhancer_config_with_fastpitch.model, fastpitch=fastpitch)


@pytest.fixture
def sample_input(batch_size=15, max_length=1000):
    generator = torch.Generator()
    generator.manual_seed(0)

    lengths = torch.randint(max_length // 4, max_length - 7, (batch_size, 1), generator=generator)
    condition = torch.randn((batch_size, 1, 80, 1000), generator=generator)

    return condition, lengths


@pytest.mark.unit
def test_pad_spectrogram(enhancer, sample_input):
    condition, lengths = sample_input
    output = enhancer.pad_spectrogram(condition)

    assert output.size(-1) >= condition.size(-1)


@pytest.mark.unit
def test_forward_pass_keeps_size(enhancer, sample_input):
    condition, lengths = sample_input
    output = enhancer.forward(condition=condition, lengths=lengths)

    assert output.shape == condition.shape


@pytest.mark.unit
def test_nemo_save_load(enhancer: SpectrogramEnhancerModel, tmp_path):
    path = tmp_path / "test-enhancer-save-load.nemo"

    enhancer.save_to(path)
    return SpectrogramEnhancerModel.restore_from(path)


@pytest.mark.with_downloads
@pytest.mark.unit
def test_nemo_save_load_with_fastpitch(enhancer_with_fastpitch: SpectrogramEnhancerModel, tmp_path):
    path = tmp_path / "test-enhancer-save-load.nemo"

    enhancer_with_fastpitch.save_to(path)
    restored_enhancer = SpectrogramEnhancerModel.restore_from(path)
    assert restored_enhancer.fastpitch is None


@pytest.mark.with_downloads
@pytest.mark.unit
def test_fastpitch_loads_data_inside_enhancer(enhancer_with_fastpitch):
    assert len(enhancer_with_fastpitch._train_dl.dataset) > 0
