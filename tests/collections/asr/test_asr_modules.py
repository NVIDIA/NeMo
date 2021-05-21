# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr import modules
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import config_utils


class TestASRModulesBasicTests:
    def compare_features(self, data1, len1, data2, len2, eps_mean_err=1e-3, eps_max_err=1e-2):
        padded_shape = torch.Size([d1 if d1 > d2 else d2 for d1, d2 in zip(data1.shape, data2.shape)])
        padded_data1 = torch.zeros(padded_shape)
        padded_data2 = torch.zeros(padded_shape)
        for i in range(len(data1)):
            sample_len1 = len1[i]
            sample_len2 = len2[i]
            assert sample_len1 == sample_len2
            padded_data1[i, :, :sample_len1] = data1[i, :, :sample_len1]
            padded_data2[i, :, :sample_len2] = data2[i, :, :sample_len2]
        diff = torch.abs(padded_data1 - padded_data2)
        assert torch.mean(diff) <= eps_mean_err
        assert torch.max(diff) <= eps_max_err

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor1(self):
        # Test 1 that should test the pure stft implementation as much as possible
        instance1 = modules.AudioToMelSpectrogramPreprocessor(
            dither=0, stft_conv=False, mag_power=1.0, normalize=False, preemph=0.0, log=False, pad_to=0
        )
        instance2 = modules.AudioToMelSpectrogramPreprocessor(
            dither=0, stft_conv=True, mag_power=1.0, normalize=False, preemph=0.0, log=False, pad_to=0
        )

        # Ensure that the two functions behave similarily
        for _ in range(10):
            input_signal = torch.randn(size=(4, 512))
            length = torch.randint(low=161, high=500, size=[4])
            res1, length1 = instance1(input_signal=input_signal, length=length)
            res2, length2 = instance2(input_signal=input_signal, length=length)
            self.compare_features(res1, length1, res2, length2)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_DALI_spectrogram(self):
        for win_len_sec, n_win_len, win_hop_sec, n_win_hop, n_fft, mag_power, win_func in [
            (0.02, None, 0.01, None, None, 1.0, 'hann'),
            (None, 320, None, 160, None, 1.0, 'hann'),
            (None, 320, None, 160, None, 2.0, 'hann'),
            (None, 320, None, 320, None, 1.0, 'hann'),
            (None, 320, None, 160, 512, 1.0, 'hann'),
            (None, 320, None, 160, 512, 1.0, 'hamming'),
            (None, 320, None, 160, 512, 1.0, 'blackman'),
            (None, 320, None, 160, 512, 1.0, 'bartlett'),
        ]:
            instance1 = modules.AudioToMelSpectrogramPreprocessor(
                window_size=win_len_sec,
                n_window_size=n_win_len,
                window_stride=win_hop_sec,
                n_window_stride=n_win_hop,
                n_fft=n_fft,
                mag_power=mag_power,
                window=win_func,
                dither=0,
                normalize=False,
                preemph=0.0,
                log=False,
                pad_to=0,
                use_dali=False,
            )
            instance2 = modules.AudioToMelSpectrogramPreprocessor(
                window_size=win_len_sec,
                n_window_size=n_win_len,
                window_stride=win_hop_sec,
                n_window_stride=n_win_hop,
                n_fft=n_fft,
                mag_power=mag_power,
                window=win_func,
                dither=0,
                normalize=False,
                preemph=0.0,
                log=False,
                pad_to=0,
                use_dali=True,
            )

            # Ensure that the two functions behave similarily
            for _ in range(5):
                input_signal = torch.randn(size=(4, 512))
                length = torch.randint(low=161, high=500, size=[4])
                res1, length1 = instance1(input_signal=input_signal, length=length)
                res2, length2 = instance2(input_signal=input_signal, length=length)
                self.compare_features(res1, length1, res2, length2)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_DALI_melfilterbank(self):
        for n_fft, features, lowfreq, highfreq in [
            (512, 64, 0, None),
            (512, 128, 4000, None),
            (512, 64, 0, 8000),
            (512, 64, 4000, 8000),
        ]:
            instance1 = modules.AudioToMelSpectrogramPreprocessor(
                n_fft=n_fft,
                features=features,
                lowfreq=lowfreq,
                highfreq=highfreq,
                dither=0,
                normalize=False,
                preemph=0.0,
                log=False,
                pad_to=0,
                use_dali=False,
            )
            instance2 = modules.AudioToMelSpectrogramPreprocessor(
                n_fft=n_fft,
                features=features,
                lowfreq=lowfreq,
                highfreq=highfreq,
                dither=0,
                normalize=False,
                preemph=0.0,
                log=False,
                pad_to=0,
                use_dali=True,
            )

            # Ensure that the two functions behave similarily
            for _ in range(5):
                input_signal = torch.randn(size=(4, 512))
                length = torch.randint(low=161, high=500, size=[4])
                res1, length1 = instance1(input_signal=input_signal, length=length)
                res2, length2 = instance2(input_signal=input_signal, length=length)
                self.compare_features(res1, length1, res2, length2)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_DALI_log(self):
        for log_zero_guard_type, log_zero_guard_value in [
            ("add", 2 ** -24),
            ("clamp", 2 ** -24),
            ("add", 1e-4),
            ("clamp", 1e-4),
        ]:
            instance1 = modules.AudioToMelSpectrogramPreprocessor(
                log=True,
                log_zero_guard_type=log_zero_guard_type,
                log_zero_guard_value=log_zero_guard_value,
                dither=0,
                normalize=False,
                preemph=0.0,
                pad_to=0,
                use_dali=False,
            )
            instance2 = modules.AudioToMelSpectrogramPreprocessor(
                log=True,
                log_zero_guard_type=log_zero_guard_type,
                log_zero_guard_value=log_zero_guard_value,
                dither=0,
                normalize=False,
                preemph=0.0,
                pad_to=0,
                use_dali=True,
            )

            # Ensure that the two functions behave similarily
            for _ in range(5):
                input_signal = torch.randn(size=(4, 512))
                length = torch.randint(low=161, high=500, size=[4])
                res1, length1 = instance1(input_signal=input_signal, length=length)
                res2, length2 = instance2(input_signal=input_signal, length=length)
                self.compare_features(res1, length1, res2, length2)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_DALI_normalization(self):
        for normalize_type, log in [
            ("per_feature", False),
            ("all_features", False),
            ("per_feature", True),
            ("all_features", True),
        ]:
            instance1 = modules.AudioToMelSpectrogramPreprocessor(
                log=log, normalize=normalize_type, dither=0, preemph=0.0, pad_to=0, use_dali=False
            )
            instance2 = modules.AudioToMelSpectrogramPreprocessor(
                log=log, normalize=normalize_type, dither=0, preemph=0.0, pad_to=0, use_dali=True
            )

            # Ensure that the two functions behave similarily
            for _ in range(5):
                input_signal = torch.randn(size=(4, 512))
                length = torch.randint(low=161, high=500, size=[4])
                res1, length1 = instance1(input_signal=input_signal, length=length)
                res2, length2 = instance2(input_signal=input_signal, length=length)
                self.compare_features(res1, length1, res2, length2, eps_mean_err=1e-3, eps_max_err=0.5)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_DALI_default_pipe(self):
        instance1 = modules.AudioToMelSpectrogramPreprocessor(dither=0, use_dali=False)
        instance2 = modules.AudioToMelSpectrogramPreprocessor(dither=0, use_dali=True)

        # Ensure that the two functions behave similarily
        for _ in range(5):
            input_signal = torch.randn(size=(4, 512))
            # DALI has a different border policy for the first sample of preemphasis filter.
            # In the DALI implementation:   x[0] = x[0] - preemph * x[0]
            # In the torch implementation:  x[0] = x[0]
            # TODO: Use border='zero' when available in DALI.
            # For now, just setting the first sample to 0
            input_signal[:, 0] = 0
            length = torch.randint(low=161, high=500, size=[4])
            res1, length1 = instance1(input_signal=input_signal, length=length)
            res2, length2 = instance2(input_signal=input_signal, length=length)
            self.compare_features(res1, length1, res2, length2, eps_mean_err=1e-3, eps_max_err=0.5)

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_config(self):
        # Test that dataclass matches signature of module
        result = config_utils.assert_dataclass_signature_match(
            modules.AudioToMelSpectrogramPreprocessor,
            modules.audio_preprocessing.AudioToMelSpectrogramPreprocessorConfig,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor_batch(self):
        # Test 1 that should test the pure stft implementation as much as possible
        instance1 = modules.AudioToMelSpectrogramPreprocessor(normalize="per_feature", dither=0, pad_to=0)

        # Ensure that the two functions behave similarily
        for _ in range(10):
            input_signal = torch.randn(size=(4, 512))
            length = torch.randint(low=161, high=500, size=[4])

            with torch.no_grad():
                # batch size 1
                res_instance, length_instance = [], []
                for i in range(input_signal.size(0)):
                    res_ins, length_ins = instance1(input_signal=input_signal[i : i + 1], length=length[i : i + 1])
                    res_instance.append(res_ins)
                    length_instance.append(length_ins)

                res_instance = torch.cat(res_instance, 0)
                length_instance = torch.cat(length_instance, 0)

                # batch size 4
                res_batch, length_batch = instance1(input_signal=input_signal, length=length)

            assert res_instance.shape == res_batch.shape
            assert length_instance.shape == length_batch.shape
            diff = torch.mean(torch.abs(res_instance - res_batch))
            assert diff <= 1e-3
            diff = torch.max(torch.abs(res_instance - res_batch))
            assert diff <= 1e-3

    @pytest.mark.unit
    def test_AudioToMelSpectrogramPreprocessor2(self):
        # Test 2 that should test the stft implementation as used in ASR models
        instance1 = modules.AudioToMelSpectrogramPreprocessor(dither=0, stft_conv=False)
        instance2 = modules.AudioToMelSpectrogramPreprocessor(dither=0, stft_conv=True)

        # Ensure that the two functions behave similarily
        for _ in range(5):
            input_signal = torch.randn(size=(4, 512))
            length = torch.randint(low=161, high=500, size=[4])
            res1, length1 = instance1(input_signal=input_signal, length=length)
            res2, length2 = instance2(input_signal=input_signal, length=length)
            self.compare_features(res1, length1, res2, length2, eps_mean_err=3e-3, eps_max_err=3)

    @pytest.mark.unit
    def test_SpectrogramAugmentationr(self):
        # Make sure constructor works
        instance1 = modules.SpectrogramAugmentation(freq_masks=10, time_masks=3, rect_masks=3)
        assert isinstance(instance1, modules.SpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])
        res0 = instance0(input_signal=input_signal, length=length)
        res = instance1(input_spec=res0[0])

        assert res.shape == res0[0].shape

    @pytest.mark.unit
    def test_SpectrogramAugmentationr_config(self):
        # Test that dataclass matches signature of module
        result = config_utils.assert_dataclass_signature_match(
            modules.SpectrogramAugmentation, modules.audio_preprocessing.SpectrogramAugmentationConfig,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_CropOrPadSpectrogramAugmentation(self):
        # Make sure constructor works
        audio_length = 128
        instance1 = modules.CropOrPadSpectrogramAugmentation(audio_length=audio_length)
        assert isinstance(instance1, modules.CropOrPadSpectrogramAugmentation)

        # Make sure forward doesn't throw with expected input
        instance0 = modules.AudioToMelSpectrogramPreprocessor(dither=0)
        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])
        res0 = instance0(input_signal=input_signal, length=length)
        res, new_length = instance1(input_signal=res0[0], length=length)

        assert res.shape == torch.Size([4, 64, audio_length])
        assert all(new_length == torch.tensor([128] * 4))

    @pytest.mark.unit
    def test_CropOrPadSpectrogramAugmentation_config(self):
        # Test that dataclass matches signature of module
        result = config_utils.assert_dataclass_signature_match(
            modules.CropOrPadSpectrogramAugmentation,
            modules.audio_preprocessing.CropOrPadSpectrogramAugmentationConfig,
        )
        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_RNNTDecoder(self):
        vocab = list(range(10))
        vocab = [str(x) for x in vocab]
        vocab_size = len(vocab)

        pred_config = OmegaConf.create(
            {
                '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
                'prednet': {'pred_hidden': 32, 'pred_rnn_layers': 1,},
                'vocab_size': vocab_size,
                'blank_as_pad': True,
            }
        )

        prednet = modules.RNNTDecoder.from_config_dict(pred_config)

        # num params
        pred_hidden = pred_config.prednet.pred_hidden
        embed = (vocab_size + 1) * pred_hidden  # embedding with blank
        rnn = (
            2 * 4 * (pred_hidden * pred_hidden + pred_hidden)
        )  # (ih + hh) * (ifco gates) * (indim * hiddendim + bias)
        assert prednet.num_weights == (embed + rnn)

        # State initialization
        x_ = torch.zeros(4, dtype=torch.float32)
        states = prednet.initialize_state(x_)

        for state_i in states:
            assert state_i.dtype == x_.dtype
            assert state_i.device == x_.device
            assert state_i.shape[1] == len(x_)

        # Blank hypotheses test
        blank = vocab_size
        hyp = Hypothesis(score=0.0, y_sequence=[blank])
        cache = {}
        pred, states, _ = prednet.score_hypothesis(hyp, cache)

        assert pred.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2
        for state_i in states:
            assert state_i.dtype == pred.dtype
            assert state_i.device == pred.device
            assert state_i.shape[1] == len(pred)

        # Blank stateless predict
        g, states = prednet.predict(y=None, state=None, add_sos=False, batch_size=1)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2
        for state_i in states:
            assert state_i.dtype == g.dtype
            assert state_i.device == g.device
            assert state_i.shape[1] == len(g)

        # Blank stateful predict
        g, states2 = prednet.predict(y=None, state=states, add_sos=False, batch_size=1)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states2) == 2
        for state_i, state_j in zip(states, states2):
            assert (state_i - state_j).square().sum().sqrt() > 0.0

        # Predict with token and state
        token = torch.full([1, 1], fill_value=0, dtype=torch.long)
        g, states = prednet.predict(y=token, state=states2, add_sos=False, batch_size=None)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2

        # Predict with blank token and no state
        token = torch.full([1, 1], fill_value=blank, dtype=torch.long)
        g, states = prednet.predict(y=token, state=None, add_sos=False, batch_size=None)

        assert g.shape == torch.Size([1, 1, pred_hidden])
        assert len(states) == 2

    @pytest.mark.unit
    def test_RNNTJoint(self):
        vocab = list(range(10))
        vocab = [str(x) for x in vocab]
        vocab_size = len(vocab)

        batchsize = 4
        encoder_hidden = 64
        pred_hidden = 32
        joint_hidden = 16

        joint_cfg = OmegaConf.create(
            {
                '_target_': 'nemo.collections.asr.modules.RNNTJoint',
                'num_classes': vocab_size,
                'vocabulary': vocab,
                'jointnet': {
                    'encoder_hidden': encoder_hidden,
                    'pred_hidden': pred_hidden,
                    'joint_hidden': joint_hidden,
                    'activation': 'relu',
                },
            }
        )

        jointnet = modules.RNNTJoint.from_config_dict(joint_cfg)

        enc = torch.zeros(batchsize, encoder_hidden, 48)  # [B, D1, T]
        dec = torch.zeros(batchsize, pred_hidden, 24)  # [B, D2, U]

        # forward call test
        out = jointnet(encoder_outputs=enc, decoder_outputs=dec)
        assert out.shape == torch.Size([batchsize, 48, 24, vocab_size + 1])  # [B, T, U, V + 1]

        # joint() step test
        enc2 = enc.transpose(1, 2)  # [B, T, D1]
        dec2 = dec.transpose(1, 2)  # [B, U, D2]
        out2 = jointnet.joint(enc2, dec2)  # [B, T, U, V + 1]
        assert (out - out2).abs().sum() <= 1e-5

        # assert vocab size
        assert jointnet.num_classes_with_blank == vocab_size + 1
