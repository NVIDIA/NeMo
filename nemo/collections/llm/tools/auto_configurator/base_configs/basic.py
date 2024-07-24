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


class Basic:
    def __init__(
        self,
        name: str = None,
        version: int = None,
        size: int = None,
        measure: str = "B",
        cfg: dict = {},
    ):
        self.name = name
        self.version = version
        self.size = size
        self.measure = measure
        self.cfg = cfg
        self.num_nodes = cfg.get("num_nodes", 8)
        self.num_gpus = cfg.get("num_gpus", 8)
        self.max_steps = cfg.get("max_steps_per_run", 50)
        self.seq_length = cfg.get("seq_length", 2048)
        self.global_batch_size = cfg.get("global_batch_size", 2048)

    def model_config(self):
        None

    def optim_config(self):
        None

    def tokenizer_config(self):
        None

    def trainer_config(self):
        None

    def data_config(self):
        None
