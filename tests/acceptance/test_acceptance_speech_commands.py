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
from unittest import TestCase

import pytest
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts import AudioLabelDataset, WaveformFeaturizer, collections, parsers, perturb
from nemo.core import DeviceType

logging = nemo.logging


freq = 16000


@pytest.mark.usefixtures("neural_factory")
class TestSpeechCommandsPytorch(TestCase):
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

    @pytest.mark.acceptance
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
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.acceptance
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
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.acceptance
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
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"num_epochs": 10, "lr": 0.0003},
        )

    @pytest.mark.acceptance
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
