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

from unittest import TestCase

import pytest

from nemo.utils.lr_policies import (
    CosineAnnealing,
    PolynomialDecayAnnealing,
    PolynomialHoldDecayAnnealing,
    SquareAnnealing,
    WarmupAnnealing,
)


class TestPolicies(TestCase):
    @pytest.mark.unit
    def test_square(self):
        policy = SquareAnnealing(100)
        lr1, lr2, lr3 = (policy(1e-3, x, 0) for x in (0, 10, 20))
        self.assertTrue(lr1 >= lr2)
        self.assertTrue(lr2 >= lr3)
        self.assertTrue(lr1 - lr2 >= lr2 - lr3)

    @pytest.mark.unit
    def test_working(self):
        total_steps = 1000
        lr_policy_cls = [
            SquareAnnealing,
            CosineAnnealing,
            WarmupAnnealing,
            PolynomialDecayAnnealing,
            PolynomialHoldDecayAnnealing,
        ]
        lr_policies = [p(total_steps=total_steps) for p in lr_policy_cls]

        for step in range(1000):
            for p in lr_policies:
                assert p(1e-3, step, 0) > 0

    @pytest.mark.unit
    def test_warmup(self):
        policy = SquareAnnealing(100, warmup_ratio=0.5)
        lr1, lr2, lr3 = (policy(1e-3, x, 0) for x in (0, 50, 100))
        self.assertTrue(lr1 < lr2)
        self.assertTrue(lr2 > lr3)

    @pytest.mark.unit
    def test_warmup_hold(self):
        policy = PolynomialHoldDecayAnnealing(1000, warmup_ratio=0.25, hold_ratio=0.25, power=2)
        lr1, lr2, lr3, lr4 = (policy(1e-3, x, 0) for x in (0, 250, 500, 1000))
        self.assertTrue(lr1 < lr2)
        self.assertTrue(lr2 == lr3)
        self.assertTrue(lr4 < lr3)
        self.assertTrue(lr4 == 0.0)
