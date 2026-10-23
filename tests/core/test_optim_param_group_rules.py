# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Dict, Union

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.core.classes import ModelPT
from nemo.core.optim.param_groups import apply_param_group_rules


class Network(torch.nn.Module):
    """Embedding tied to the output projection, a norm, a convolution and a projection with bias."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.conv = torch.nn.Conv1d(8, 8, kernel_size=1)
        self.hidden = torch.nn.Linear(8, 8)
        self.output = torch.nn.Linear(8, 12, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, x):
        return self.output(self.hidden(self.conv(self.norm(self.embedding(x)).transpose(1, 2)).transpose(1, 2)))


NO_WEIGHT_DECAY = {'module_types': ['torch.nn.Embedding', 'torch.nn.LayerNorm'], 'param_names': ['*.bias']}


def _names(network: torch.nn.Module, group: Dict) -> list:
    by_id = {id(param): name for name, param in network.named_parameters()}
    return [by_id[id(param)] for param in group['params']]


class TestApplyParamGroupRules:
    @pytest.mark.unit
    def test_selection_by_type_and_name(self):
        network = Network()
        groups = apply_param_group_rules(
            network,
            [{'params': list(network.parameters()), 'lr': 0.5}],
            {
                'no_weight_decay': dict(NO_WEIGHT_DECAY, weight_decay=0.0),
                'conv_flag': {'module_types': [torch.nn.Conv1d], 'use_muon': True},
            },
        )
        # one sub-group per combination of matching rules, in the order of the first parameter of each
        assert [_names(network, group) for group in groups] == [
            ['embedding.weight', 'norm.weight', 'norm.bias', 'hidden.bias'],
            ['conv.weight'],
            ['conv.bias'],
            ['hidden.weight'],
        ]
        assert groups[0] == {'lr': 0.5, 'weight_decay': 0.0, 'params': groups[0]['params']}
        assert groups[1] == {'lr': 0.5, 'use_muon': True, 'params': groups[1]['params']}
        # the convolution bias matches both rules and takes both options
        assert groups[2] == {'lr': 0.5, 'weight_decay': 0.0, 'use_muon': True, 'params': groups[2]['params']}
        assert groups[3] == {'lr': 0.5, 'params': groups[3]['params']}
        assert sum(len(group['params']) for group in groups) == len(list(network.parameters()))

    @pytest.mark.unit
    def test_shared_parameter_matches_through_any_owner_or_name(self):
        network = Network()
        tied = [network.output.weight]
        by_type = apply_param_group_rules(
            network, [{'params': tied}], {'rule': {'module_types': ['torch.nn.Embedding'], 'weight_decay': 0.0}}
        )
        assert by_type == [{'weight_decay': 0.0, 'params': tied}]
        by_name = apply_param_group_rules(
            network, [{'params': tied}], {'rule': {'param_names': ['output.weight'], 'weight_decay': 0.0}}
        )
        assert by_name == [{'weight_decay': 0.0, 'params': tied}]
        unmatched = apply_param_group_rules(
            network, [{'params': tied}], {'rule': {'param_names': ['hidden.*'], 'weight_decay': 0.0}}
        )
        assert unmatched == [{'params': tied}]

    @pytest.mark.unit
    def test_several_base_groups_and_rule_order(self):
        network = Network()
        groups = apply_param_group_rules(
            network,
            [
                {'params': [network.hidden.weight, network.hidden.bias], 'lr': 1.0},
                {'params': [network.norm.weight], 'lr': 2.0, 'weight_decay': 0.3},
            ],
            {
                'first': {'param_names': ['*'], 'weight_decay': 0.1, 'momentum': 0.8},
                'second': {'param_names': ['*.bias', 'norm.*'], 'weight_decay': 0.0},
            },
        )
        assert groups == [
            {'lr': 1.0, 'weight_decay': 0.1, 'momentum': 0.8, 'params': [network.hidden.weight]},
            {'lr': 1.0, 'weight_decay': 0.0, 'momentum': 0.8, 'params': [network.hidden.bias]},
            {'lr': 2.0, 'weight_decay': 0.0, 'momentum': 0.8, 'params': [network.norm.weight]},
        ]

    @pytest.mark.unit
    def test_invalid_rules(self):
        network = Network()
        groups = [{'params': list(network.parameters())}]
        with pytest.raises(ValueError, match="selects nothing"):
            apply_param_group_rules(network, groups, {'empty': {'weight_decay': 0.0}})
        with pytest.raises(ValueError, match="sets no parameter group options"):
            apply_param_group_rules(network, groups, {'noop': {'param_names': ['*']}})
        with pytest.raises(ValueError, match="must not set `params`"):
            apply_param_group_rules(network, groups, {'bad': {'param_names': ['*'], 'params': []}})
        with pytest.raises(TypeError, match="not a torch.nn.Module class"):
            apply_param_group_rules(network, groups, {'bad': {'module_types': ['torch.nn.functional.relu'], 'lr': 1}})
        assert apply_param_group_rules(network, groups, {}) == groups


class RuledModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.network = Network()

    def forward(self, x):
        return self.network(x)

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = None

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = None

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = None

    @classmethod
    def list_available_models(cls):
        return []


class TestModelPTParamGroupRules:
    @pytest.mark.unit
    def test_rules_split_the_optimizer_groups(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 0.01},
                'optim_param_group_rules': {'no_weight_decay': dict(NO_WEIGHT_DECAY, weight_decay=0.0)},
            }
        )
        model = RuledModel(cfg)
        model.setup_optimization()
        groups = model._optimizer.param_groups
        assert [group['weight_decay'] for group in groups] == [0.0, 0.01]
        assert [_names(model.network, group) for group in groups] == [
            ['embedding.weight', 'norm.weight', 'norm.bias', 'conv.bias', 'hidden.bias'],
            ['conv.weight', 'hidden.weight'],
        ]
        assert all(group['lr'] == 1e-3 for group in groups)

    @pytest.mark.unit
    def test_rules_apply_within_attribute_groups(self):
        cfg = OmegaConf.create(
            {
                'optim': {'name': 'sgd', 'lr': 0.1},
                'optim_param_groups': {'network': {'lr': 0.2}},
                'optim_param_group_rules': {'biases': {'param_names': ['*.bias'], 'lr': 0.3}},
            }
        )
        model = RuledModel(cfg)
        model.setup_optimization()
        groups = model._optimizer.param_groups
        assert [(group['lr'], _names(model.network, group)) for group in groups] == [
            (0.2, ['embedding.weight', 'norm.weight', 'conv.weight', 'hidden.weight']),
            (0.3, ['norm.bias', 'conv.bias', 'hidden.bias']),
        ]

    @pytest.mark.unit
    def test_without_rules_nothing_changes(self):
        cfg = OmegaConf.create({'optim': {'name': 'sgd', 'lr': 0.1}})
        model = RuledModel(cfg)
        model.setup_optimization()
        assert len(model._optimizer.param_groups) == 1
        assert len(model._optimizer.param_groups[0]['params']) == len(list(model.parameters()))
