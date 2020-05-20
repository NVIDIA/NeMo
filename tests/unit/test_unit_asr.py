#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import os
import shutil
import tarfile
import unittest
from unittest import TestCase

import pytest
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts import AudioDataset, WaveformFeaturizer, collections, parsers
from nemo.core import DeviceType
from nemo.utils import logging

freq = 16000


@pytest.mark.usefixtures("neural_factory")
class TestUnitASRPytorch(TestCase):
    labels = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]
    manifest_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/asr/an4_train.json"))
    featurizer_config = {
        'window': 'hann',
        'dither': 1e-05,
        'normalize': 'per_feature',
        'frame_splicing': 1,
        'int_values': False,
        'window_stride': 0.01,
        'sample_rate': freq,
        'features': 64,
        'n_fft': 512,
        'window_size': 0.02,
    }
    yaml = YAML(typ="safe")

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        logging.info("Looking up for test ASR data")
        if not os.path.exists(os.path.join(data_folder, "asr")):
            logging.info("Extracting ASR data to: {0}".format(os.path.join(data_folder, "asr")))
            tar = tarfile.open(os.path.join(data_folder, "asr.tar.gz"), "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            logging.info("ASR data found in: {0}".format(os.path.join(data_folder, "asr")))

    @pytest.mark.unit
    def test_transcript_normalizers(self):
        # Create test json
        test_strings = [
            "TEST CAPITALIZATION",
            '!\\"#$%&\'()*+,-./:;<=>?@[\\\\]^_`{|}~',
            "3+3=10",
            "3 + 3 = 10",
            "why     is \\t whitepsace\\tsuch a problem   why indeed",
            "\\\"Can you handle quotes?,\\\" says the boy",
            "I Jump!!!!With joy?Now.",
            "Maybe I want to learn periods.",
            "$10 10.90 1-800-000-0000",
            "18000000000 one thousand 2020",
            "1 10 100 1000 10000 100000 1000000",
            "Î  ĻƠvɆȩȅĘ ÀÁÃ Ą ÇĊňńŤŧș",
            "‘’“”❛❜❝❞「 」 〈 〉 《 》 【 】 〔 〕 ⦗ ⦘ 😙  👀 🔨",
            "It only costs $1 000 000! Cheap right?",
            "2500, 3000 are separate but 200, 125 is not",
            "1",
            "1 2",
            "1 2 3",
            "10:00pm is 10:00 pm is 22:00 but not 10: 00 pm",
            "10:00 10:01pm 10:10am 10:90pm",
            "Mr. Expand me!",
            "Mr Don't Expand me!",
        ]
        normalized_strings = [
            "test capitalization",
            'percent and \' plus',
            "three plus three ten",
            "three plus three ten",
            "why is whitepsace such a problem why indeed",
            "can you handle quotes says the boy",
            "i jump with joy now",
            "maybe i want to learn periods",
            "ten dollars ten point nine zero one eight hundred zero zero",
            "eighteen billion one thousand two thousand and twenty",
            # Two line string below
            "one ten thousand one hundred one thousand ten thousand one hundred thousand one million",
            "i loveeee aaa a ccnntts",
            "''",
            "it only costs one million dollars cheap right",
            # Two line string below
            "two thousand five hundred three thousand are separate but two "
            "hundred thousand one hundred and twenty five is not",
            "one",
            "one two",
            "one two three",
            "ten pm is ten pm is twenty two but not ten zero pm",
            "ten ten one pm ten ten am ten ninety pm",
            "mister expand me",
            "mr don't expand me",
        ]
        manifest_paths = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/asr/manifest_test.json"))

        def remove_test_json():
            os.remove(manifest_paths)

        self.addCleanup(remove_test_json)

        with open(manifest_paths, "w") as f:
            for s in test_strings:
                f.write('{"audio_filepath": "", "duration": 1.0, "text": ' f'"{s}"}}\n')
        parser = parsers.make_parser(self.labels, 'en')
        manifest = collections.ASRAudioText(manifests_files=[manifest_paths], parser=parser,)

        for i, s in enumerate(normalized_strings):
            self.assertTrue(manifest[i].text_tokens == parser(s))

    @pytest.mark.unit
    def test_pytorch_audio_dataset(self):
        featurizer = WaveformFeaturizer.from_config(self.featurizer_config)
        ds = AudioDataset(manifest_filepath=self.manifest_filepath, labels=self.labels, featurizer=featurizer,)

        for i in range(len(ds)):
            if i == 5:
                logging.info(ds[i])
            # logging.info(ds[i][0].shape)
            # self.assertEqual(freq, ds[i][0].shape[0])

    @pytest.mark.unit
    def test_dataloader(self):
        batch_size = 4
        dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
        )
        for ind, data in enumerate(dl.data_iterator):
            # With num_workers update, this is no longer true
            # Moving to GPU is handled by AudioPreprocessor
            # data is on GPU
            # self.assertTrue(data[0].is_cuda)
            # self.assertTrue(data[1].is_cuda)
            # self.assertTrue(data[2].is_cuda)
            # self.assertTrue(data[3].is_cuda)
            # first dimension is batch
            self.assertTrue(data[0].size(0) == batch_size)
            self.assertTrue(data[1].size(0) == batch_size)
            self.assertTrue(data[2].size(0) == batch_size)
            self.assertTrue(data[3].size(0) == batch_size)

    @pytest.mark.unit
    def test_preprocessor_errors(self):
        def create_broken_preprocessor_1():
            nemo_asr.AudioToMelSpectrogramPreprocessor(window_size=2, n_window_size=2)

        def create_broken_preprocessor_2():
            nemo_asr.AudioToMelSpectrogramPreprocessor(window_stride=2, n_window_stride=2)

        def create_broken_preprocessor_3():
            nemo_asr.AudioToMelSpectrogramPreprocessor(n_window_stride=2)

        def create_good_preprocessor_1():
            nemo_asr.AudioToMelSpectrogramPreprocessor(window_size=0.02, window_stride=0.01)

        def create_good_preprocessor_2():
            nemo_asr.AudioToMelSpectrogramPreprocessor(
                window_size=None, window_stride=None, n_window_size=256, n_window_stride=32,
            )

        self.assertRaises(ValueError, create_broken_preprocessor_1)
        self.assertRaises(ValueError, create_broken_preprocessor_2)
        self.assertRaises(ValueError, create_broken_preprocessor_3)
        create_good_preprocessor_1()
        create_good_preprocessor_2()

    @pytest.mark.unit
    def test_kaldi_dataloader(self):
        batch_size = 4
        dl = nemo_asr.KaldiFeatureDataLayer(
            kaldi_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/kaldi_an4/')),
            labels=self.labels,
            batch_size=batch_size,
        )
        for data in dl.data_iterator:
            self.assertTrue(data[0].size(0) == batch_size)

        dl_test_min = nemo_asr.KaldiFeatureDataLayer(
            kaldi_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/kaldi_an4/')),
            labels=self.labels,
            batch_size=batch_size,
            min_duration=1.0,
        )
        self.assertTrue(len(dl_test_min) == 18)

        dl_test_max = nemo_asr.KaldiFeatureDataLayer(
            kaldi_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/kaldi_an4/')),
            labels=self.labels,
            batch_size=batch_size,
            max_duration=5.0,
        )
        self.assertTrue(len(dl_test_max) == 19)

    @pytest.mark.unit
    def test_tarred_dataloader(self):
        batch_size = 4
        manifest_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../data/asr/tarred_an4/tarred_audio_manifest.json')
        )

        # Test loading a single tarball
        tarpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/tarred_an4/audio_0.tar'))
        dl_single_tar = nemo_asr.TarredAudioToTextDataLayer(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, batch_size=batch_size
        )
        count = 0
        for _ in dl_single_tar.dataset:
            count += 1
        self.assertTrue(count == 16)

        # Test braceexpand loading
        tarpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/asr/tarred_an4/audio_{0..3}.tar'))
        dl_braceexpand = nemo_asr.TarredAudioToTextDataLayer(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, batch_size=batch_size
        )
        self.assertTrue(len(dl_braceexpand) == 65)
        count = 0
        for _ in dl_braceexpand.dataset:
            count += 1
        self.assertTrue(count == 65)

        # Test loading via list
        tarpath = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), f'../data/asr/tarred_an4/audio_{i}.tar'))
            for i in range(4)
        ]
        dl_list_load = nemo_asr.TarredAudioToTextDataLayer(
            audio_tar_filepaths=tarpath, manifest_filepath=manifest_path, labels=self.labels, batch_size=batch_size
        )
        count = 0
        for _ in dl_braceexpand.dataset:
            count += 1
        self.assertTrue(count == 65)

    @pytest.mark.unit
    def test_trim_silence(self):
        batch_size = 4
        normal_dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
            shuffle=False,
        )
        trimmed_dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            trim_silence=True,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
            shuffle=False,
        )
        for norm, trim in zip(normal_dl.data_iterator, trimmed_dl.data_iterator):
            for point in range(batch_size):
                self.assertTrue(norm[1][point].data >= trim[1][point].data)

    @pytest.mark.unit
    def test_audio_preprocessors(self):
        batch_size = 5
        dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=True,
            shuffle=False,
        )

        installed_torchaudio = True
        try:
            import torchaudio
        except ModuleNotFoundError:
            installed_torchaudio = False
            with self.assertRaises(ModuleNotFoundError):
                to_spectrogram = nemo_asr.AudioToSpectrogramPreprocessor(n_fft=400, window=None)
            with self.assertRaises(ModuleNotFoundError):
                to_mfcc = nemo_asr.AudioToMFCCPreprocessor(n_mfcc=15)

        if installed_torchaudio:
            to_spectrogram = nemo_asr.AudioToSpectrogramPreprocessor(n_fft=400, window=None)
            to_mfcc = nemo_asr.AudioToMFCCPreprocessor(n_mfcc=15)

        to_melspec = nemo_asr.AudioToMelSpectrogramPreprocessor(features=50)

        for batch in dl.data_iterator:
            input_signals, seq_lengths, _, _ = batch
            input_signals = input_signals.to(to_melspec._device)
            seq_lengths = seq_lengths.to(to_melspec._device)

            melspec = to_melspec.forward(input_signals, seq_lengths)

            if installed_torchaudio:
                spec = to_spectrogram.forward(input_signals, seq_lengths)
                mfcc = to_mfcc.forward(input_signals, seq_lengths)

            # Check that number of features is what we expect
            self.assertTrue(melspec[0].shape[1] == 50)

            if installed_torchaudio:
                self.assertTrue(spec[0].shape[1] == 201)  # n_fft // 2 + 1 bins
                self.assertTrue(mfcc[0].shape[1] == 15)
