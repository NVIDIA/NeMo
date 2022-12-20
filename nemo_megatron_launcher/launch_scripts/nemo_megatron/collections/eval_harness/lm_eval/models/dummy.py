# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import random

from lm_eval.base import LM


class DummyLM(LM):
    def __init__(self):
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string):
        return cls()

    def loglikelihood(self, requests):
        res = []

        for _ in requests:
            res.append((-random.random(), False))

        return res

    def greedy_until(self, requests):
        res = []

        for ctx, _ in requests:
            res.append("lol")
            assert ctx.strip() != ""

        return res

    def loglikelihood_rolling(self, requests):
        res = []

        for _ in requests:
            res.append(-random.random())

        return res
