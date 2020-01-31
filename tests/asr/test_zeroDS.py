# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2019 NVIDIA. All Rights Reserved.
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

import torch
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.core.neural_types import *
from tests.common_setup import NeMoUnitTest


class TestZeroDL(NeMoUnitTest):
    labels = [
        "'",
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
        " ",
    ]
    manifest_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/asr/an4_train.json"))
    yaml = YAML(typ="safe")

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        print("Looking up for test ASR data")
        if not os.path.exists(os.path.join(data_folder, "asr")):
            print("Extracting ASR data to: {0}".format(os.path.join(data_folder, "asr")))
            tar = tarfile.open(os.path.join(data_folder, "asr.tar.gz"), "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            print("ASR data found in: {0}".format(os.path.join(data_folder, "asr")))

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
        print("Looking up for test ASR data")
        if os.path.exists(os.path.join(data_folder, "asr")):
            shutil.rmtree(os.path.join(data_folder, "asr"))

    def test_simple_train(self):
        print("Simplest train test with ZeroDL")
        neural_factory = nemo.core.neural_factory.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, create_tb_writer=False
        )
        trainable_module = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        data_source = nemo.backends.pytorch.common.ZerosDataLayer(
            size=10000,
            dtype=torch.FloatTensor,
            batch_size=128,
            output_ports={
                "x": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag, dim=1)}),
                "y": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag, dim=1)}),
            },
        )
        loss = nemo.backends.pytorch.tutorials.MSELoss()
        x, y = data_source()
        y_pred = trainable_module(x=x)
        loss_tensor = loss(predictions=y_pred, target=y)

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss_tensor], print_func=lambda x: print(f'Train Loss: {str(x[0].item())}')
        )
        neural_factory.train(
            [loss_tensor], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd"
        )

    def test_asr_with_zero_ds(self):
        print("Testing ASR NMs with ZeroDS and without pre-processing")
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/jasper_smaller.yaml"))
        with open(path) as file:
            jasper_model_definition = self.yaml.load(file)

        dl = nemo.backends.pytorch.common.ZerosDataLayer(
            size=100,
            dtype=torch.FloatTensor,
            batch_size=4,
            output_ports={
                "processed_signal": NeuralType(
                    {
                        0: AxisType(BatchTag),
                        1: AxisType(SpectrogramSignalTag, dim=64),
                        2: AxisType(ProcessedTimeTag, dim=64),
                    }
                ),
                "processed_length": NeuralType({0: AxisType(BatchTag)}),
                "transcript": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag, dim=64)}),
                "transcript_length": NeuralType({0: AxisType(BatchTag)}),
            },
        )

        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition["JasperEncoder"],
        )
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))

        # DAG
        processed_signal, p_length, transcript, transcript_len = dl()
        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # print(jasper_encoder)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ctc_loss(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len
        )

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: print(f'Train Loss: {str(x[0].item())}')
        )
        # Instantiate an optimizer to perform `train` action
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch, local_rank=None, create_tb_writer=False
        )
        neural_factory.train(
            [loss], callbacks=[callback], optimization_params={"num_epochs": 2, "lr": 0.0003}, optimizer="sgd"
        )
