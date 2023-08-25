# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import typing

import numpy as np
from pytriton.client import ModelClient

from .utils import str_list2numpy


class NemoQuery:
    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name

    def query_llm(self, prompts, max_output_len=200, init_timeout=600.0):
        prompts = str_list2numpy(prompts)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(prompts=prompts, max_output_len=max_output_len)
            output_type = client.model_config.outputs[0].dtype

        if output_type == np.bytes_:
            sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
            return sentences
        else:
            return result_dict["outputs"]
