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
from typing import Dict
from unittest import TestCase

import numpy as np
import pytest
import torch
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.collections.nlp.nm.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.nm.trainables.common import TokenClassifier
from nemo.core import WeightShareTransform
from nemo.core.neural_types import *
from nemo.utils import logging


@pytest.mark.usefixtures("neural_factory")
class TestWeightSharing(TestCase):
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
    manifest_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/asr/an4_train.json"))
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
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/"))
        logging.info("Looking up for test ASR data")
        if not os.path.exists(os.path.join(data_folder, "asr")):
            logging.info("Extracting ASR data to: {0}".format(os.path.join(data_folder, "asr")))
            tar = tarfile.open(os.path.join(data_folder, "asr.tar.gz"), "r:gz")
            tar.extractall(path=data_folder)
            tar.close()
        else:
            logging.info("ASR data found in: {0}".format(os.path.join(data_folder, "asr")))

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     super().tearDownClass()
    #     data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
    #     logging.info("Looking up for test ASR data")
    #     if os.path.exists(os.path.join(data_folder, "asr")):
    #         logging.info("Removing test ASR data")
    #         shutil.rmtree(os.path.join(data_folder, "asr"))

    def __check_if_weights_are_equal(self, w1: Dict, w2: Dict):
        all_same = set(w1.keys()) == set(w2.keys())
        if not all_same:
            return False
        else:
            for key in w1.keys():
                all_same = all_same and np.array_equal(
                    w1[key][0].cpu().detach().numpy(), w2[key][0].cpu().detach().numpy(),
                )
        return all_same

    @pytest.mark.unit
    def test_TaylorNet_get_weights(self):
        tn1 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        tn2 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        # because of randomness, actual weights should be different
        self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(), tn2.get_weights()))
        tn3 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
        tn3.set_weights(tn1.get_weights())
        # check than weights are the same
        self.assertTrue(self.__check_if_weights_are_equal(tn1.get_weights(), tn3.get_weights()))
        # change weights on one module - another module should not change
        tn1.fc1.bias.data = torch.tensor([0.1])
        self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(), tn3.get_weights()))

    # def test_TaylorNet_tie_weights(self):
    #     tn1 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
    #     tn2 = nemo.backends.pytorch.tutorials.TaylorNet(dim=4)
    #     # because of randomness, actual weights should be different
    #     self.assertFalse(self.__check_if_weights_are_equal(tn1.get_weights(), tn2.get_weights()))
    #     tn2.tie_weights_with(tn1, list(tn1.get_weights().keys()))
    #     # change weights on one module - another module should change too
    #     tn2.fc1.bias.data = torch.tensor([0.1])
    #     self.assertTrue(self.__check_if_weights_are_equal(tn1.get_weights(), tn2.get_weights()))

    @pytest.mark.unit
    def test_tie_weights(self):
        class DummyDataLayer(DataLayerNM):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size

                class DummyDS(torch.utils.data.Dataset):
                    def __init__(self, vocab_size):
                        super().__init__()

                    def __getitem__(self, index):
                        model_inputs = torch.randint(high=vocab_size, size=[10])
                        model_outputs = torch.randint(high=vocab_size, size=[10])
                        return (model_inputs, model_outputs)

                    def __len__(self):
                        return 10

                self._dataset = DummyDS(vocab_size)

            @property
            def output_ports(self):
                return {
                    "model_inputs": NeuralType(('B', 'T')),
                    "model_outputs": NeuralType(('B', 'T'), LabelsType()),
                }

            def __len__(self):
                return len(self._dataset)

            @property
            def dataset(self):
                return self._dataset

            def data_iterator(self):
                pass

        voc_size = 10
        dim = 10
        embd = nemo.backends.pytorch.common.other.SequenceEmbedding(voc_size=voc_size, hidden_size=dim)
        proj = TokenClassifier(hidden_size=dim, num_classes=voc_size)
        data = DummyDataLayer(voc_size)
        loss = SmoothedCrossEntropyLoss(pad_id=0)
        embd.tie_weights_with(
            proj,
            weight_names=["embedding.weight"],
            name2name_and_transform={"embedding.weight": ("mlp.layer2.weight", WeightShareTransform.SAME)},
        )
        self.assertTrue(
            np.array_equal(embd.embedding.weight.detach().cpu().numpy(), proj.mlp.layer2.weight.detach().cpu().numpy())
        )
        _in, _out = data()
        pred = embd(input_seq=_in)
        pred = proj(hidden_states=pred)
        loss_t = loss(labels=_out, logits=pred)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss_t], optimizer="sgd", optimization_params={"max_steps": 5, "lr": 0.0003},
        )

        self.assertTrue(
            np.array_equal(embd.embedding.weight.detach().cpu().numpy(), proj.mlp.layer2.weight.detach().cpu().numpy())
        )

    @pytest.mark.unit
    def test_untied_weights(self):
        class DummyDataLayer(DataLayerNM):
            def __init__(self, vocab_size):
                super().__init__()
                self.vocab_size = vocab_size

                class DummyDS(torch.utils.data.Dataset):
                    def __init__(self, vocab_size):
                        super().__init__()

                    def __getitem__(self, index):
                        model_inputs = torch.randint(high=vocab_size, size=[10])
                        model_outputs = torch.randint(high=vocab_size, size=[10])
                        return (model_inputs, model_outputs)

                    def __len__(self):
                        return 10

                self._dataset = DummyDS(vocab_size)

            @property
            def output_ports(self):
                return {
                    "model_inputs": NeuralType(('B', 'T')),
                    "model_outputs": NeuralType(('B', 'T'), LabelsType()),
                }

            def __len__(self):
                return len(self._dataset)

            @property
            def dataset(self):
                return self._dataset

            def data_iterator(self):
                pass

        voc_size = 10
        dim = 10
        embd = nemo.backends.pytorch.common.other.SequenceEmbedding(voc_size=voc_size, hidden_size=dim)
        proj = TokenClassifier(hidden_size=dim, num_classes=voc_size)
        data = DummyDataLayer(voc_size)
        loss = SmoothedCrossEntropyLoss(pad_id=0)
        # embd.tie_weights_with(
        #     proj,
        #     weight_names=["embedding.weight"],
        #     name2name_and_transform={"embedding.weight": ("mlp.layer2.weight", WeightShareTransform.SAME)},
        # )
        self.assertFalse(
            np.array_equal(embd.embedding.weight.detach().cpu().numpy(), proj.mlp.layer2.weight.detach().cpu().numpy())
        )
        _in, _out = data()
        pred = embd(input_seq=_in)
        pred = proj(hidden_states=pred)
        loss_t = loss(labels=_out, logits=pred)

        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss_t], optimizer="sgd", optimization_params={"max_steps": 5, "lr": 0.0003},
        )

        self.assertFalse(
            np.array_equal(embd.embedding.weight.detach().cpu().numpy(), proj.mlp.layer2.weight.detach().cpu().numpy())
        )

    @pytest.mark.unit
    def test_set_weights(self):
        voc_size = 3
        dim = 2
        embd = nemo.backends.pytorch.common.SequenceEmbedding(voc_size=voc_size, hidden_size=dim)
        weights = torch.tensor(np.random.randint(0, 10, (3, 2)) * 1.0)
        name2weights = {"embedding.weight": (weights, True)}
        embd.set_weights(name2weight=name2weights)
        self.assertTrue(np.array_equal(embd.embedding.weight.detach().cpu().numpy(), weights.detach().cpu().numpy()))
        weights = torch.tensor(np.random.randint(0, 10, (3, 2)) * 1.0)
        self.assertFalse(np.array_equal(embd.embedding.weight.detach().cpu().numpy(), weights.detach().cpu().numpy()))

    @pytest.mark.unit
    @pytest.mark.run_only_on('GPU')
    def test_freeze_unfreeze_TrainableNM(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/jasper_smaller.yaml"))
        with open(path) as file:
            jasper_model_definition = self.yaml.load(file)
        dl = nemo_asr.AudioToTextDataLayer(
            # featurizer_config=self.featurizer_config,
            manifest_filepath=self.manifest_filepath,
            labels=self.labels,
            batch_size=4,
        )
        pre_process_params = {
            #'int_values': False,
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
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder'],
        )
        jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))
        ctc_loss = nemo_asr.CTCLossNM(num_classes=len(self.labels))
        jasper_encoder.freeze()
        jasper_encoder.unfreeze(set(['encoder.4.mconv.0.conv.weight']))
        frozen_weight = jasper_encoder.encoder[1].mconv[0].conv.weight.detach().cpu().numpy()
        unfrozen_weight = jasper_encoder.encoder[4].mconv[0].conv.weight.detach().cpu().numpy()
        # jasper_decoder.unfreeze()
        # DAG
        audio_signal, a_sig_length, transcript, transcript_len = dl()
        processed_signal, p_length = preprocessing(input_signal=audio_signal, length=a_sig_length)

        encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=p_length)
        # logging.info(jasper_encoder)
        log_probs = jasper_decoder(encoder_output=encoded)
        loss = ctc_loss(
            log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len,
        )

        callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
        )
        optimizer = nemo.backends.pytorch.actions.PtActions()
        optimizer.train(
            [loss], callbacks=[callback], optimizer="sgd", optimization_params={"max_steps": 5, "lr": 0.0003},
        )
        new_frozen_weight = jasper_encoder.encoder[1].mconv[0].conv.weight.data
        new_unfrozen_weight = jasper_encoder.encoder[4].mconv[0].conv.weight.data
        self.assertTrue(np.array_equal(frozen_weight, new_frozen_weight.detach().cpu().numpy()))
        self.assertFalse(np.array_equal(unfrozen_weight, new_unfrozen_weight.detach().cpu().numpy()))
