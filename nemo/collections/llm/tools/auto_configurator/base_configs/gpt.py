# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections import llm
from nemo.collections.llm.utils import Config

from .basic import Basic


class GPT(Basic):
    def __init__(
        self,
        model: Config = None,
        cfg: dict = {},
    ):
        """
        Args:
            name (str): model name.
            cfg (dict): auto configurator runner config.
        """

        super().__init__(model=model, cfg=cfg)

    def get_model_config(self) -> Config:
        """Function that returns model config.

        Returns:
            Config: model config.
        """

        self.model.global_batch_size = self.global_batch_size
        self.model.seq_length = self.seq_length

        return self.model
