# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron_t5 import MegatronT5Model
from nemo.utils import logging

__all__ = ["MegatronBARTModel"]


class MegatronBARTModel(MegatronT5Model):
    """
    Megatron BART pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        if self._cfg.data.get('dataset_type', None) != 'bart':
            raise ValueError(f"cfg.data.dataset_type = {self._cfg.data.get('dataset_type', None)} but 'bart' is expected")

        self.build_train_valid_test_datasets_kwargs.update(dict(
            delete_mask_prob=self._cfg.data.get('delete_mask_prob', 0.0),
        ))

    def _build_vocab(self):
        self._add_special_tokens_to_tokenizer()

        super()._build_vocab()

    def list_available_models(self):
        pass
