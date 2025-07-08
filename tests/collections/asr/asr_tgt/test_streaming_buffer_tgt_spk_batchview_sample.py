import math
import os

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_tgt_spk_models import EncDecHybridRNNTCTCTgtSpkBPEModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig
from nemo.collections.asr.parts.utils.streaming_tgt_spk_audio_buffer_ctc_batchview_sample_utils import (
    AudioBufferer_tgt_spk,
    AudioBuffersDatalayer_tgt_spk,
    AudioIterator_tgt_spk,
    FrameBatchASR_tgt_spk,
)


@pytest.fixture()
def hybrid_asr_tgt_spk_model(test_data_dir):
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    model_defaults = {'enc_hidden': 512, 'pred_hidden': 64}

    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
        'feat_in': 64,
        'feat_out': -1,
        'n_layers': 17,
        'd_model': 512,
        'use_bias': True,
        'subsampling': 'dw_striding',
        'subsampling_factor': 8,
        'subsampling_conv_channels': 256,
        'ff_expansion_factor': 4,
        'self_attention_model': 'rel_pos',
        'n_heads': 8,
        'att_context_size': [-1, -1],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {
            'pred_hidden': model_defaults['pred_hidden'],
            'pred_rnn_layers': 1,
        },
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {
            'joint_hidden': 32,
            'activation': 'relu',
        },
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 30}}

    tokenizer = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'}

    loss = {'loss_name': 'default', 'warprnnt_numba_kwargs': {'fastemit_lambda': 0.001}}

    aux_ctc = {
        'ctc_loss_weight': 0.3,
        'use_cer': False,
        'ctc_reduction': 'mean_batch',
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
            'feat_in': 512,
            'num_classes': -2,
            'vocabulary': None,
        },
        'decoding': DictConfig(CTCBPEDecodingConfig),
    }

    modelConfig = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'tokenizer': DictConfig(tokenizer),
            'decoding': DictConfig(decoding),
            'loss': DictConfig(loss),
            'aux_ctc': DictConfig(aux_ctc),
            'sample_rate': 16000,
        }
    )

    model_instance = EncDecHybridRNNTCTCTgtSpkBPEModel(cfg=modelConfig)
    model_instance.change_decoding_strategy(modelConfig.decoding, decoder_type='ctc')
    return model_instance


@pytest.mark.unit
def test_audio_iterator_tgt_spk():
    # Create sample audio data
    samples = np.random.randn(16000).astype('float32')  # 1 second of audio
    query_samples = np.random.randn(8000).astype('float32')  # 0.5 seconds of audio
    frame_len = 0.1  # 100ms frame
    device = torch.device('cpu')

    iterator = AudioIterator_tgt_spk(samples, query_samples, frame_len, device)

    # Test iterator properties
    assert iterator._feature_frame_len == int(frame_len * 16000)
    assert iterator.audio_signal.shape == (1, 16000)
    assert iterator.query_audio_signal.shape == (1, 8000)

    # Test iteration
    frame = next(iterator)
    assert frame.shape == (1, int(frame_len * 16000))


@pytest.mark.unit
def test_audio_bufferer_tgt_spk():
    class MockASRModel:
        def __init__(self):
            self._cfg = type('obj', (object,), {'sample_rate': 16000})
            self.preprocessor = type('obj', (object,), {'log': False})

    asr_model = MockASRModel()
    frame_len = 0.1
    batch_size = 2
    total_buffer = 0.3

    bufferer = AudioBufferer_tgt_spk(
        asr_model=asr_model, frame_len=frame_len, batch_size=batch_size, total_buffer=total_buffer
    )

    # Test initialization
    assert bufferer.feature_frame_len == int(frame_len * 16000)
    assert bufferer.batch_size == batch_size
    assert bufferer.buffer.shape == (1, int(total_buffer * 16000))


@pytest.mark.unit
def test_audio_buffers_datalayer_tgt_spk():
    datalayer = AudioBuffersDatalayer_tgt_spk()

    # Set test signal
    test_signal = [np.random.randn(1000).astype('float32') for _ in range(3)]
    datalayer.set_signal(test_signal)

    # Test iteration
    signal, length = next(datalayer)
    assert isinstance(signal, torch.Tensor)
    assert isinstance(length, torch.Tensor)
    assert signal.shape == (1000,)
    assert length.item() == 1000


@pytest.mark.parametrize(
    "frame_len, total_buffer, expected_tokens_per_chunk, expected_mid_delay, tokens_per_buffer",
    [
        (4.8, 8, 60, 80, 100),
        (1, 4, 13, 32, 50),
        # (0.16, 4, 2, 26, 50),
    ],
)
@pytest.mark.unit
def test_frame_batch_asr_tgt_spk(
    hybrid_asr_tgt_spk_model, frame_len, total_buffer, expected_tokens_per_chunk, expected_mid_delay, tokens_per_buffer
):
    hybrid_asr_tgt_spk_model.eval()
    batch_size = 2
    model_stride_in_secs = 0.08
    tokens_per_chunk = math.ceil(frame_len / model_stride_in_secs)
    mid_delay = math.ceil((frame_len + (total_buffer - frame_len) / 2) / model_stride_in_secs)

    assert tokens_per_chunk == expected_tokens_per_chunk
    assert mid_delay == expected_mid_delay

    frame_asr = FrameBatchASR_tgt_spk(
        asr_model=hybrid_asr_tgt_spk_model, frame_len=frame_len, total_buffer=total_buffer, batch_size=batch_size
    )

    # Test initialization
    assert frame_asr.frame_len == frame_len
    assert frame_asr.batch_size == batch_size
    assert isinstance(frame_asr.frame_bufferer, AudioBufferer_tgt_spk)

    # Test reset
    frame_asr.reset()
    assert len(frame_asr.unmerged) == 0
    assert len(frame_asr.all_logits) == 0
    assert len(frame_asr.all_preds) == 0

    # Test read_audio_file
    json_line = {
        "audio_filepath": "/home/TestData/an4_diarizer/simulated_train/multispeaker_session_0.wav",
        "offset": 0.007,
        "duration": 14.046,
        "text": "illustration the lemon various dishes are frequently ordered add the wine and if necessary a seasoning of cayenne when it will be rain",
        "num_speakers": 2,
        "rttm_filepath": "/home/TestData/an4_diarizer/simulated_train/multispeaker_session_0.1919_1988.rttm",
        "uem_filepath": None,
        "ctm_filepath": None,
        "query_speaker_id": "1919",
        "query_audio_filepath": "/home/TestData/an4_diarizer/simulated_train/multispeaker_session_0.wav",
        "query_offset": 0.007,
        "query_duration": 4,
    }

    json_line['duration'] = 5

    frame_asr.read_audio_file(
        audio_filepath=json_line['audio_filepath'],
        offset=json_line['offset'],
        duration=json_line['duration'],
        query_audio_file=json_line['query_audio_filepath'],
        query_offset=json_line['query_offset'],
        query_duration=json_line['query_duration'],
        separater_freq=500,
        separater_duration=0.1,
        separater_unvoice_ratio=0.3,
        delay=mid_delay,
        model_stride_in_secs=model_stride_in_secs,
    )

    frame_asr.tokens_per_chunk = tokens_per_chunk
    frame_asr.delay = mid_delay
    frame_asr.infer_logits()
    assert len(frame_asr.all_preds[0]) == tokens_per_buffer
    hyp = frame_asr.transcribe(tokens_per_chunk, mid_delay)
