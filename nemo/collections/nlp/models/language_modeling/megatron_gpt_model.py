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

from megatron.global_vars import get_args
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_megatron_for_nemo
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from megatron.model import GPTModel
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.utils import logging


class MegatronGPTModel(NLPModel):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        initialize_megatron_for_nemo(
            micro_batch_size=cfg.get('micro_batch_size', 1),
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            encoder_seq_length=cfg.get('encoder_seq_length', 512),
            num_layers=cfg.get('num_layers', 1),
            hidden_size=cfg.get('hidden_size', 16),
            num_attention_heads=cfg.get('num_attention_heads', 1),
            max_position_embeddings=cfg.get('max_position_embeddings', 512),
        )
        args = get_args()
        vars(args)['padded_vocab_size'] = 10  # temporarily set so we can instaniate GPTModel

        self.model = GPTModel(
            num_tokentypes=0, parallel_output=True, pre_process=cfg.pre_process, post_process=cfg.post_process
        )

        logging.info(f'done with constructor')

    def forward(self):

        pass

    def setup_training_data(self, cfg):
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=cfg.data_prefix,
            data_impl=cfg.data_impl,
            splits_string=cfg.splits_string,
            train_valid_test_num_samples=cfg.train_valid_test_num_samples,
            seq_length=cfg.seq_length,
            seed=cfg.seed,
            skip_warmup=cfg.skip_warmup,
        )
        logging.info(f'done with setup_training_data')

    def setup_validation_data(self, cfg):
        pass

    def list_available_models():
        pass

