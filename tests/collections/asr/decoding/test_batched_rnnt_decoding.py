# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from functools import lru_cache

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint
from nemo.collections.asr.parts.mixins import mixins
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as greedy_decode
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import Best1BeamBatchedInfer
from nemo.collections.asr.parts.submodules.tdt_beam_decoding import Best1BeamBatchedTDTInfer
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig, RNNTBPEDecoding, RNNTDecoding
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


def char_vocabulary():
    return [' ', 'a', 'b', 'c', 'd', 'e', 'f', '.']


@pytest.fixture()
@lru_cache(maxsize=8)
def tmp_tokenizer(test_data_dir):
    cfg = DictConfig({'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'})

    class _TmpASRBPE(mixins.ASRBPEMixin):
        def register_artifact(self, _, vocab_path):
            return vocab_path

    asrbpe = _TmpASRBPE()
    asrbpe._setup_tokenizer(cfg)
    return asrbpe.tokenizer


@lru_cache(maxsize=2)
def get_rnnt_decoder(vocab_size, decoder_output_size=4):
    prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
    torch.manual_seed(0)
    decoder = RNNTDecoder(prednet=prednet_cfg, vocab_size=vocab_size)
    decoder.freeze()
    return decoder


@lru_cache(maxsize=2)
def get_rnnt_joint(vocab_size, vocabulary=None, encoder_output_size=4, decoder_output_size=4, joint_output_shape=4):
    jointnet_cfg = {
        'encoder_hidden': encoder_output_size,
        'pred_hidden': decoder_output_size,
        'joint_hidden': joint_output_shape,
        'activation': 'relu',
    }
    torch.manual_seed(0)
    joint = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=vocabulary)
    joint.freeze()
    return joint


@lru_cache(maxsize=1)
def get_model_encoder_output(data_dir, model_name):
    # Import inside function to avoid issues with dependencies
    import librosa

    audio_filepath = os.path.join(data_dir, 'asr', 'test', 'an4', 'wav', 'cen3-fjlp-b.wav')

    with torch.no_grad():
        model = ASRModel.from_pretrained(model_name)  # type: ASRModel
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()

        audio, sr = librosa.load(path=audio_filepath, sr=16000, mono=True)

        input_signal = torch.tensor(audio, dtype=torch.float32, device=model.device).unsqueeze(0)
        input_signal_length = torch.tensor([len(audio)], dtype=torch.int32, device=model.device)

        encoded, encoded_len = model(input_signal=input_signal, input_signal_length=input_signal_length)

    return model, encoded, encoded_len


def decode_text_from_greedy_hypotheses(hyps, decoding):
    decoded_hyps = decoding.decode_hypothesis(hyps)  # type: List[str]

    return decoded_hyps


def decode_text_from_nbest_hypotheses(hyps, decoding):
    hypotheses = []
    all_hypotheses = []

    for nbest_hyp in hyps:  # type: rnnt_utils.NBestHypotheses
        n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
        decoded_hyps = decoding.decode_hypothesis(n_hyps)  # type: List[str]

        hypotheses.append(decoded_hyps[0])  # best hypothesis
        all_hypotheses.append(decoded_hyps)

    return hypotheses, all_hypotheses

def check_beam_decoding_rnnt(test_data_dir, beam_config):
    beam_size = beam_config.pop("beam_size", 1)
    model, encoded, encoded_len = get_model_encoder_output(test_data_dir, 'nvidia/stt_en_fastconformer_transducer_large')
    print("Device: ", encoded.device)

    beam = rnnt_beam_decoding.Best1BeamBatchedInfer(
        model.decoder,
        model.joint,
        beam_size=beam_size,
        return_best_hypothesis=False,
        **beam_config,
    )

    enc_out = encoded
    enc_len = encoded_len

    with torch.no_grad():
        hyps: rnnt_utils.Hypothesis = beam(encoder_output=enc_out, encoded_lengths=enc_len)[0]
        _, all_hyps = decode_text_from_nbest_hypotheses(hyps, model.decoding)
        all_hyps = all_hyps[0]

        print("Beam search algorithm :", beam_config['search_type'])
        for idx, hyp_ in enumerate(all_hyps):
            print("Hyp index", idx + 1, "text :", hyp_.text)

            assert len(hyp_.timestamp) > 0
            print("Timesteps", hyp_.timestamp)
            print()

# def check_beam_decoding_rnnt(test_data_dir, beam_config):
#     beam_size = beam_config.pop("beam_size", 1)
#     model, encoded, encoded_len = get_model_encoder_output(test_data_dir, 'nvidia/stt_en_fastconformer_tdt_large')

#     model_config = model.to_config_dict()
#     durations = list(model_config["model_defaults"]["tdt_durations"])

#     beam = tdt_beam_decoding.BeamTDTInfer(
#         model.decoder,
#         model.joint,
#         beam_size=beam_size,
#         return_best_hypothesis=False,
#         durations=durations,
#         **beam_config,
#     )

#     enc_out = encoded
#     enc_len = encoded_len

#     with torch.no_grad():
#         hyps: rnnt_utils.Hypothesis = beam(encoder_output=enc_out, encoded_lengths=enc_len)[0]
#         _, all_hyps = decode_text_from_nbest_hypotheses(hyps, model.decoding)
#         all_hyps = all_hyps[0]

#         print("Beam search algorithm :", beam_config['search_type'])
#         for idx, hyp_ in enumerate(all_hyps):
#             print("Hyp index", idx + 1, "text :", hyp_.text)

#             assert len(hyp_.timestamp) > 0
#             print("Timesteps", hyp_.timestamp)
#             print()


class TestRNNTDecoding:
#     @pytest.mark.skipif(
#         not NUMBA_RNNT_LOSS_AVAILABLE,
#         reason='RNNTLoss has not been compiled with appropriate numba version.',
#     )
#     @pytest.mark.with_downloads
#     @pytest.mark.unit
#     @pytest.mark.parametrize(
#         "beam_config",
#         [
#             {"search_type": "malsd_batch", "beam_size": 2, "allow_cuda_graphs": False},
#             {"search_type": "malsd_batch", "beam_size": 4, "allow_cuda_graphs": False},
#             {"search_type": "malsd_batch", "beam_size": 2},
#             {"search_type": "malsd_batch", "beam_size": 4},
#             {"search_type": "maes_batch", "beam_size": 2},
#             {"search_type": "maes_batch", "beam_size": 4},
#         ]
#     )
#     def test_rnnt_beam_decoding_return_best_hypothesis(self, test_data_dir, beam_config):
#         beam_size = beam_config.pop("beam_size", 1)
#         model, encoder_output, encoded_lengths = get_model_encoder_output(test_data_dir, 'stt_en_conformer_transducer_small')
#         vocab_size = model.tokenizer.vocab_size
#         decoding = Best1BeamBatchedInfer(
#             model.decoder,
#             model.joint,
#             blank_index=vocab_size,
#             beam_size=beam_size,
#             score_norm=True,
#             return_best_hypothesis=True,
#             **beam_config,
#         )
        
#         with torch.no_grad():
#             hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]
#             assert type(hyps) == list
#             assert type(hyps[0]) == rnnt_utils.Hypothesis
            
#             assert len(hyps) == 1
#             assert hasattr(hyps[0], "y_sequence")
#             assert hasattr(hyps[0], "score")
#             assert hasattr(hyps[0], "timestamp")
            
#             assert len(hyps[0].y_sequence) > 0
#             assert len(hyps[0].timestamp) > 0
            
#             hyps = decode_text_from_greedy_hypotheses(hyps, model.decoding)
            
#             print()
            
#             print(f"Beam search algorithm : {beam_config['search_type']}, beam size: {beam_size}")
#             print("Decoded text: ", hyps[0].text)
#             print("Score: ", hyps[0].score)
#             print("Transcript", hyps[0].y_sequence)
#             print("Timesteps", hyps[0].timestamp)
#             print()
          
            
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "beam_size": 2, "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "beam_size": 4, "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "beam_size": 2},
            {"search_type": "malsd_batch", "beam_size": 4},
            {"search_type": "maes_batch", "beam_size": 2},
            {"search_type": "maes_batch", "beam_size": 4},
            {"search_type": "maes_batch", "beam_size": 4, "maes_expansion_gamma": 2, "maes_expansion_beta": 4},
        ]
    )
    def test_rnnt_beam_decoding_return_nbest(self, test_data_dir, beam_config):
        beam_size = beam_config.pop("beam_size", 1)
        model, encoder_output, encoded_lengths = get_model_encoder_output(test_data_dir, 'stt_en_conformer_transducer_small')
        vocab_size = model.tokenizer.vocab_size
        decoding = Best1BeamBatchedInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=False,
            **beam_config,
        )
        
        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            assert type(hyps) == list
            assert type(hyps[0]) == rnnt_utils.NBestHypotheses
            
            assert len(hyps) == 1
            
            assert hasattr(hyps[0].n_best_hypotheses[0], "y_sequence")
            assert hasattr(hyps[0].n_best_hypotheses[0], "score")
            assert hasattr(hyps[0].n_best_hypotheses[0], "timestamp")
            
            assert len(hyps[0].n_best_hypotheses[0].y_sequence) > 0
            assert len(hyps[0].n_best_hypotheses[0].timestamp) > 0
            
            _, all_hyps = decode_text_from_nbest_hypotheses(hyps, model.decoding)
            all_hyps = all_hyps[0]

            print()
            print(f"Beam search algorithm : {beam_config['search_type']}, beam size: {beam_size}")
            for idx, hyp_ in enumerate(all_hyps):
                print("Hyp index", idx + 1, "text :", hyp_.text)
                print("Score: ", hyp_.score)

                assert len(hyp_.timestamp) > 0
                print("Transcripts: ", hyp_.y_sequence)
                print("Timesteps: ", hyp_.timestamp)
                print()