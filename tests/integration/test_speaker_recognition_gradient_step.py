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

import pytest
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr

logging = nemo.logging


@pytest.mark.usefixtures("neural_factory")
class TestSpeakerRecognitonPytorch:
    manifest_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/an4_speaker/train.json"))
    yaml = YAML(typ="safe")

    @classmethod
    def setup_class(cls) -> None:
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        logging.info("Looking up for speaker related data")
        if not os.path.exists(os.path.join(data_folder, "an4_speaker")):
            logging.info("Extracting speaker related files to: {0}".format(os.path.join(data_folder, "an4_speaker")))
            tar = tarfile.open(os.path.join(data_folder, "an4_speaker.tar.gz"), "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            logging.info("Speech Command data found in: {0}".format(os.path.join(data_folder, "an4_speaker")))

    @classmethod
    def teardown_class(cls) -> None:
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        logging.info("Looking up for test an4 data")
        if os.path.exists(os.path.join(data_folder, "an4_speaker")):
            shutil.rmtree(os.path.join(data_folder, "an4_speaker"))

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
    def test_quartznet_speaker_reco_training(self):
        """Integtaion test that instantiates a small QuartzNet model for speaker recognition and tests training with the
        sample an4 data.
        Training is run for 3 forward and backward steps and asserts that loss after 3 steps is smaller than the loss
        at the first step.
        """
        with open(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/quartznet_spkr_test.yaml"))
        ) as file:
            spkr_params = self.yaml.load(file)
        dl = nemo_asr.AudioToSpeechLabelDataLayer(manifest_filepath=self.manifest_filepath, labels=None, batch_size=5,)
        sample_rate = 16000

        preprocessing = nemo_asr.AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate, **spkr_params["AudioToMelSpectrogramPreprocessor"],
        )
        jasper_encoder = nemo_asr.JasperEncoder(**spkr_params['JasperEncoder'])
        jasper_decoder = nemo_asr.JasperDecoderForSpkrClass(
            feat_in=spkr_params['JasperEncoder']['jasper'][-1]['filters'],
            num_classes=dl.num_classes,
            pool_mode=spkr_params['JasperDecoderForSpkrClass']['pool_mode'],
            emb_sizes=spkr_params["JasperDecoderForSpkrClass"]["emb_sizes"].split(","),
        )
        ce_loss = nemo_asr.CrossEntropyLossNM()

        # DAG
        audio_signal, a_sig_length, targets, targets_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        log_probs, _ = jasper_decoder(encoder_output=encoded)
        loss = ce_loss(logits=log_probs, labels=targets)

        loss_list = []
        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=partial(self.print_and_log_loss, loss_log_list=loss_list), step_freq=1
        )
        self.nf.random_seed = 42
        self.nf.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"max_steps": 4, "lr": 0.002},
        )
        self.nf.reset_trainer()

        # Assert that training loss went down
        assert loss_list[-1] < loss_list[0]
