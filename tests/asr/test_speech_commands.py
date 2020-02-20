# ! /usr/bin/python
# -*- coding: utf-8 -*-

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

from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts import AudioLabelDataset, WaveformFeaturizer, collections, parsers, perturb
from nemo.core import DeviceType
from tests.common_setup import NeMoUnitTest

logging = nemo.logging


freq = 16000


class TestSpeechCommandsPytorch(NeMoUnitTest):
    labels = [
        "cat",
        "dog",
    ]
    manifest_filepath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/speech_commands/train_manifest.json")
    )
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
        logging.info("Looking up for test speech command data")
        if not os.path.exists(os.path.join(data_folder, "speech_commands")):
            logging.info(
                "Extracting speech commands data to: {0}".format(os.path.join(data_folder, "speech_commands"))
            )
            tar = tarfile.open(os.path.join(data_folder, "speech_commands.tar.xz"), "r:xz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            logging.info("Speech Command data found in: {0}".format(os.path.join(data_folder, "speech_commands")))

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        logging.info("Looking up for test ASR data")
        if os.path.exists(os.path.join(data_folder, "speech_commands")):
            shutil.rmtree(os.path.join(data_folder, "speech_commands"))

    def test_pytorch_audio_dataset_with_perturbation(self):
        perturbations = [
            perturb.WhiteNoisePerturbation(min_level=-90, max_level=-46),
            perturb.ShiftPerturbation(min_shift_ms=-5.0, max_shift_ms=5.0),
        ]

        # Execute perturbations with 100% probability
        prob_perturb = [(1.0, p) for p in perturbations]

        audio_augmentor = perturb.AudioAugmentor(prob_perturb)

        featurizer = WaveformFeaturizer(
            sample_rate=self.featurizer_config['sample_rate'],
            int_values=self.featurizer_config['int_values'],
            augmentor=audio_augmentor,
        )
        ds = AudioLabelDataset(manifest_filepath=self.manifest_filepath, labels=self.labels, featurizer=featurizer,)

        for i in range(len(ds)):
            logging.info(ds[i])
            # logging.info(ds[i][0].shape)
            # self.assertEqual(freq, ds[i][0].shape[0])

    def test_dataloader(self):
        batch_size = 2
        dl = nemo_asr.AudioToSpeechLabelDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            sample_rate=16000,
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

    def test_trim_silence(self):
        batch_size = 2
        normal_dl = nemo_asr.AudioToSpeechLabelDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=False,
            shuffle=False,
        )
        trimmed_dl = nemo_asr.AudioToSpeechLabelDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            trim_silence=True,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=False,
            shuffle=False,
        )
        for norm, trim in zip(normal_dl.data_iterator, trimmed_dl.data_iterator):
            for point in range(batch_size):
                self.assertTrue(norm[1][point].data >= trim[1][point].data)

    def test_audio_preprocessors(self):
        batch_size = 2
        dl = nemo_asr.AudioToSpeechLabelDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=batch_size,
            # placement=DeviceType.GPU,
            drop_last=False,
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

    # @unittest.skip("Init parameters of nemo_asr.AudioToMelSpectrogramPreprocessor are invalid")
    def test_jasper_training(self):
        with open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_speech_recognition.yaml"))
        ) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToSpeechLabelDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=2,
        )
        pre_process_params = pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(**jasper_model_definition['JasperEncoder'],)
        jasper_decoder = nemo_asr.JasperDecoderForClassification(
            feat_in=jasper_model_definition['JasperEncoder']['jasper'][-1]['filters'], num_classes=len(self.labels)
        )
        ce_loss = nemo_asr.CrossEntropyLossNM()

        # DAG
        audio_signal, a_sig_length, targets, targets_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ce_loss(logits=log_probs, labels=targets)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    # @unittest.skip("Init parameters of nemo_asr.AudioToMelSpectrogramPreprocessor are invalid")
    def test_double_jasper_training(self):
        with open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_speech_recognition.yaml"))
        ) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToSpeechLabelDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=2,
        )
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder1 = nemo_asr.JasperEncoder(**jasper_model_definition['JasperEncoder'],)
        jasper_encoder2 = nemo_asr.JasperEncoder(**jasper_model_definition['JasperEncoder'],)
        # mx_max1 = nemo.backends.pytorch.common.other.SimpleCombiner(mode="max")
        # mx_max2 = nemo.backends.pytorch.common.other.SimpleCombiner(mode="max")
        jasper_decoder1 = nemo_asr.JasperDecoderForClassification(
            feat_in=jasper_model_definition['JasperEncoder']['jasper'][-1]['filters'], num_classes=len(self.labels)
        )
        jasper_decoder2 = nemo_asr.JasperDecoderForClassification(
            feat_in=jasper_model_definition['JasperEncoder']['jasper'][-1]['filters'], num_classes=len(self.labels)
        )

        ce_loss = nemo_asr.CrossEntropyLossNM()

        # DAG
        audio_signal, a_sig_length, targets, targets_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded1, encoded_len1 = jasper_encoder1(audio_signal=processed_signal, length=p_length)
        encoded2, encoded_len2 = jasper_encoder2(audio_signal=processed_signal, length=p_length)
        logits1 = jasper_decoder1(encoder_output=encoded1)
        logits2 = jasper_decoder2(encoder_output=encoded2)
        loss = ce_loss(logits=logits1, labels=targets,)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(str(x[0].item()))
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    def test_stft_conv(self):
        with open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_speech_recognition.yaml"))
        ) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToSpeechLabelDataLayer(
            manifest_filepath=self.manifest_filepath, labels=self.labels, batch_size=2,
        )
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
            'stft_conv': True,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(**jasper_model_definition['JasperEncoder'],)
        jasper_decoder = nemo_asr.JasperDecoderForClassification(
            feat_in=jasper_model_definition['JasperEncoder']['jasper'][-1]['filters'], num_classes=len(self.labels)
        )

        ce_loss = nemo_asr.CrossEntropyLossNM()

        # DAG
        audio_signal, a_sig_length, targets, targets_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        logits = jasper_decoder(encoder_output=encoded)
        loss = ce_loss(logits=logits, labels=targets)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(str(x[0].item()))
        )
        # Instantiate an optimizer to perform `train` action
        optimizer = self.nf.get_trainer()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    def test_jasper_eval(self):
        with open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_speech_recognition.yaml"))
        ) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToSpeechLabelDataLayer(
            manifest_filepath=self.manifest_filepath, labels=self.labels, batch_size=2,
        )
        pre_process_params = {
            'frame_splicing': 1,
            'features': 64,
            'window_size': 0.02,
            'n_fft': 512,
            'dither': 1e-05,
            'window': 'hann',
            'sample_rate': 16000,
            'normalize': 'per_feature',
            'window_stride': 0.01,
        }
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(**jasper_model_definition['JasperEncoder'],)
        jasper_decoder = nemo_asr.JasperDecoderForClassification(
            feat_in=jasper_model_definition['JasperEncoder']['jasper'][-1]['filters'], num_classes=len(self.labels)
        )
        ce_loss = nemo_asr.CrossEntropyLossNM()

        # DAG
        audio_signal, a_sig_length, targets, targets_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        logits = jasper_decoder(encoder_output=encoded)
        loss = ce_loss(logits=logits, labels=targets,)

        from nemo.collections.asr.helpers import (
            process_classification_evaluation_batch,
            process_classification_evaluation_epoch,
        )

        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[loss, logits, targets],
            user_iter_callback=lambda x, y: process_classification_evaluation_batch(x, y, top_k=[1]),
            user_epochs_done_callback=process_classification_evaluation_epoch,
        )
        # Instantiate an optimizer to perform `train` action
        self.nf.eval(callbacks=[eval_callback])
