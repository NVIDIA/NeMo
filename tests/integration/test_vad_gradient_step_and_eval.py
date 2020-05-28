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
from functools import partial
from unittest import TestCase

import pytest
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import logging


@pytest.mark.usefixtures("neural_factory")
class TestVADPytorch(TestCase):
    labels = [
        "background",
        "speech",
    ]
    manifest_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/vad/train_manifest.json"))
    featurizer_config = {
        'window': 'hann',
        'dither': 1e-05,
        'normalize': 'per_feature',
        'frame_splicing': 1,
        'int_values': False,
        'window_stride': 0.01,
        'sample_rate': 16000,
        'features': 64,
        'n_fft': 512,
        'window_size': 0.02,
    }
    yaml = YAML(typ="safe")

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        logging.info(data_folder)
        logging.info("Looking up for test vad data")
        if not os.path.exists(os.path.join(data_folder, "vad")):
            logging.info("Extracting vad data to: {0}".format(os.path.join(data_folder, "vad")))
            tar = tarfile.open(os.path.join(data_folder, "vad.tar.xz"), "r:xz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            logging.info("VAD data found in: {0}".format(os.path.join(data_folder, "vad")))

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        logging.info("Looking up for test vad data")
        if os.path.exists(os.path.join(data_folder, "vad")):
            shutil.rmtree(os.path.join(data_folder, "vad"))

    @staticmethod
    def print_and_log_loss(loss_tensor, loss_log_list):
        """A helper function that is passed to SimpleLossLoggerCallback. It prints loss_tensors and appends to
        the loss_log_list list.

        Args:
            loss_tensor (NMTensor): tensor representing loss. Loss should be a scalar
            loss_log_list (list): empty list
        """
        logging.info(f'Train Loss: {str(loss_tensor[0].item())}')
        loss_log_list.append(loss_tensor[0].item())

    @pytest.mark.integration
    def test_quartznet_vad_training(self):
        """Integtaion test that instantiates a small QuartzNet model for vad and tests training with the
        sample vad data.
        Training is run for 3 forward and backward steps and asserts that loss after 3 steps is smaller than the loss
        at the first step.
        """
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_vad.yaml"))) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToSpeechLabelDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=6,
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
        #         preprocessing = nemo_asr.AudioToMFCCPreprocessor(**pre_process_params)
        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(**pre_process_params)
        jasper_encoder = nemo_asr.JasperEncoder(**jasper_model_definition['JasperEncoder'])
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

        loss_list = []
        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=partial(self.print_and_log_loss, loss_log_list=loss_list), step_freq=1
        )

        self.nf.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"max_steps": 3, "lr": 0.003},
        )
        self.nf.reset_trainer()

        # Assert that training loss went down
        assert loss_list[-1] < loss_list[0]

    @pytest.mark.integration
    def test_quartznet_vad_eval(self):
        """Integration test that tests EvaluatorCallback and NeuralModuleFactory.eval().
        """
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_vad.yaml"))) as file:
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
        jasper_encoder = nemo_asr.JasperEncoder(**jasper_model_definition['JasperEncoder'])
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
