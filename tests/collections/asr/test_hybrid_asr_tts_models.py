from pathlib import Path

import pytest
from omegaconf import DictConfig

from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.models.hybrid_asr_tts_models import ASRWithTTSModel
from nemo.collections.tts.models import FastPitchModel


@pytest.fixture
def fastpitch_model():
    model = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch_multispeaker")
    return model


@pytest.fixture
def fastpitch_model_path(fastpitch_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("tts_models") / "fastpitch.nemo"
    fastpitch_model.save_to(path)
    return path


@pytest.fixture
def conformer_ctc_bpe_bn_model():
    model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    return model


@pytest.fixture
def conformer_ctc_bpe_bn_model_path(conformer_ctc_bpe_bn_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("asr_models") / "conformer-ctc-bpe-bn.nemo"
    conformer_ctc_bpe_bn_model.save_to(path)
    return path


@pytest.fixture
def conformer_rnnt_bpe_bn_model():
    model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_small")
    return model


@pytest.fixture
def conformer_rnnt_bpe_bn_model_path(conformer_rnnt_bpe_bn_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("asr_models") / "conformer-rnnt-bpe.nemo"
    conformer_rnnt_bpe_bn_model.save_to(path)
    return path


@pytest.fixture
def asr_model_ctc_bpe_config(test_data_dir):
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 1024,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 1024,
        'num_classes': -1,
        'vocabulary': None,
    }

    tokenizer = {'dir': str(Path(test_data_dir) / "asr/tokenizers/an4_wpe_128"), 'type': 'wpe'}

    model_config = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'tokenizer': DictConfig(tokenizer),
        }
    )
    return model_config


class TestASRWithTTSModel:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_from_pretrained_models(self, fastpitch_model_path, conformer_ctc_bpe_bn_model_path):
        model = ASRWithTTSModel.from_pretrained_models(
            asr_model_path=conformer_ctc_bpe_bn_model_path, tts_model_path=fastpitch_model_path
        )
        assert isinstance(model.tts_model, FastPitchModel)
        assert isinstance(model.asr_model, EncDecCTCModelBPE)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_from_asr_config(self, asr_model_ctc_bpe_config, fastpitch_model_path):
        model = ASRWithTTSModel.from_asr_config(
            asr_cfg=asr_model_ctc_bpe_config,
            asr_model_type="ctc_bpe",
            tts_model_path=fastpitch_model_path,
        )
        assert isinstance(model.tts_model, FastPitchModel)
        assert isinstance(model.asr_model, EncDecCTCModelBPE)
