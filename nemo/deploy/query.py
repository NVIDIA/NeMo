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

import typing
import numpy as np
import logging
from pytriton.client import ModelClient


def str_list2numpy(str_list: typing.List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


class NemoQuery:

    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name

    def query_gpt(self, prompts, init_timeout=600.0, verbose=False):
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
        logger = logging.getLogger("nemo.client")

        prompts = str_list2numpy(prompts)

        logger.info("================================")
        logger.info("Preparing the client")
        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            logger.info("================================")
            logger.info("Sent batch for inference:")
            result_dict = client.infer_batch(prompts=prompts)
            return result_dict["outputs"]

