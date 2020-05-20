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
from unittest import TestCase

import pytest
import torch

import nemo
from nemo.backends.pytorch.common import DataCombination
from nemo.core import ChannelType, NeuralType
from nemo.utils import logging


@pytest.mark.usefixtures("neural_factory")
class TestMultiDLUnit(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    @pytest.mark.unit
    def test_port_name_collision_handling(self):
        batch_size = 4
        dataset_size = 4
        shuffle = False
        dl_1 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "b": NeuralType(('B', 'T'), ChannelType())},
        )
        dl_2 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "c": NeuralType(('B', 'T'), ChannelType())},
        )

        data_layer = nemo.backends.pytorch.common.MultiDataLayer(
            data_layers=[dl_1, dl_2],
            batch_size=batch_size,
            shuffle=shuffle,
            combination_mode=DataCombination.CROSSPRODUCT,
        )
        self.assertEqual([*data_layer.output_ports], ["a", "b", "a_1", "c"])
        self.assertEqual(len(data_layer), dataset_size * dataset_size)

    @pytest.mark.unit
    def test_port_renaming(self):
        batch_size = 4
        dataset_size = 4
        shuffle = False
        dl_1 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "b": NeuralType(('B', 'T'), ChannelType())},
        )
        dl_2 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "b": NeuralType(('B', 'T'), ChannelType())},
        )

        data_layer = nemo.backends.pytorch.common.MultiDataLayer(
            data_layers=[dl_1, dl_2],
            batch_size=batch_size,
            shuffle=shuffle,
            combination_mode=DataCombination.CROSSPRODUCT,
            port_names=["1", "2", "3", "4"],
        )
        self.assertEqual([*data_layer.output_ports], ["1", "2", "3", "4"])

    @pytest.mark.unit
    def test_multi_dl_zip_working(self):
        dataset_size_0 = 2
        dataset_size_1 = 2
        final_dataset_size = 2
        batch_size = 4
        shuffle = False
        dl_1 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size_0,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "b": NeuralType(('B', 'T'), ChannelType())},
        )
        dl_2 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size_1,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "c": NeuralType(('B', 'T'), ChannelType())},
        )

        data_layer = nemo.backends.pytorch.common.MultiDataLayer(
            data_layers=[dl_1, dl_2], batch_size=batch_size, shuffle=shuffle, combination_mode=DataCombination.ZIP
        )
        self.assertEqual(len(data_layer), final_dataset_size)

    @pytest.mark.unit
    def test_multi_dl_zip_failing(self):
        dataset_size_0 = 4
        dataset_size_1 = 2
        batch_size = 4
        shuffle = False
        dl_1 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size_0,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "b": NeuralType(('B', 'T'), ChannelType())},
        )
        dl_2 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size_1,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "c": NeuralType(('B', 'T'), ChannelType())},
        )

        with pytest.raises(ValueError):
            data_layer = nemo.backends.pytorch.common.MultiDataLayer(
                data_layers=[dl_1, dl_2], batch_size=batch_size, shuffle=shuffle, combination_mode=DataCombination.ZIP
            )

    @pytest.mark.unit
    def test_multi_dl_wrong_combination(self):
        dataset_size_0 = 2
        dataset_size_1 = 2
        unknown_combination = "cross"
        batch_size = 4
        shuffle = False
        dl_1 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size_0,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "b": NeuralType(('B', 'T'), ChannelType())},
        )
        dl_2 = nemo.backends.pytorch.common.ZerosDataLayer(
            size=dataset_size_1,
            dtype=torch.FloatTensor,
            batch_size=batch_size,
            output_ports={"a": NeuralType(('B', 'T'), ChannelType()), "c": NeuralType(('B', 'T'), ChannelType())},
        )

        with pytest.raises(ValueError):
            data_layer = nemo.backends.pytorch.common.MultiDataLayer(
                data_layers=[dl_1, dl_2], batch_size=batch_size, shuffle=shuffle, combination_mode=unknown_combination
            )
