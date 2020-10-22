# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import hashlib
import json
import os
from nemo.utils import app_state
from typing import List

import torch
from megatron import mpu
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.parallel import DistributedDataParallel
from transformers import TRANSFORMERS_CACHE

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.modules import BertModule, MegatronBertEncoder
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes import ModelPT
from nemo.utils import AppState, logging

__all__ = ['NLPModel']

NEMO_NLP_TMP = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), "nemo_nlp_tmp")

os.makedirs(NEMO_NLP_TMP, exist_ok=True)


class NLPModel(ModelPT):
    """Base class for NLP Models.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        self.bert_model = None  # Pretrained BERT encoder

        self.set_world_size(trainer)

    @rank_zero_only
    def register_bert_model(self):
        """Adds encoder config to .nemo archive.
        """
        # check if there is an encoder, warn if not
        if self.bert_model is None:
            raise ValueError('Instantiate self.bert_model before registering it.')
        else:
            # get encoder config and create source for artifact
            if isinstance(self.bert_model, MegatronBertEncoder):
                pretrained_model_name = self.bert_model._model_name
                encoder_config_path = pretrained_model_name + '_encoder_config.json'
                encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path)
                config_for_json = OmegaConf.to_container(self.bert_model.config)
                with open(encoder_config_src, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(config_for_json, indent=2, sort_keys=True) + '\n')
                self.register_artifact(encoder_config_path, encoder_config_src)
                self.cfg.language_model.config_file = encoder_config_path
            elif isinstance(self.bert_model, BertModule):
                # HuggingFace Transformer Config
                pretrained_model_name = self.bert_model.name_or_path
                # Some HF names have "/" in them so we replace with _
                pretrained_model_name = pretrained_model_name.replace("/", "_")
                encoder_config_path = pretrained_model_name + '_encoder_config.json'
                encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path)
                self.bert_model.config.to_json_file(encoder_config_src)  # name requested by jarvis team
                self.register_artifact(encoder_config_path, encoder_config_src)
                self.cfg.language_model.config_file = encoder_config_path
            else:
                logging.info(
                    f'Registering BERT model config for {self.bert_model} is not yet supported. Please override this method if needed.'
                )

    def setup_tokenizer(self, cfg: DictConfig):
        """Instantiates tokenizer based on config and registers tokenizer artifacts.

           If model is being restored from .nemo file then the tokenizer.vocab_file will 
           be used (if it exists). 

           Otherwise, we will use the vocab file provided in the config (if it exists).

           Finally, if no vocab file is given (this happens frequently when using HF), 
           we will attempt to extract the vocab from the tokenizer object and then register it.

        Args:
            cfg (DictConfig): Tokenizer config
        """

        if self._is_model_being_restored() and os.path.exists('tokenizer.vocab_file'):
            # model is being restored from .nemo file so tokenizer.vocab_file has precedence
            vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src='tokenizer.vocab_file')
        elif cfg.vocab_file is not None:
            # use vocab file from config
            vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file)
        else:
            vocab_file = None

        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            vocab_file=vocab_file,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            tokenizer_model=self.register_artifact(config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model),
        )
        self.tokenizer = tokenizer

        if vocab_file is None:
            # when there is no vocab file we try to get the vocab from the tokenizer and register it
            self._register_vocab_from_tokenizer(vocab_file_config_path='tokenizer.vocab_file', cfg=cfg)

    @rank_zero_only
    def _register_vocab_from_tokenizer(
        self,
        vocab_file_config_path: str = 'tokenizer.vocab_file',
        vocab_dict_config_path: str = 'tokenizer_vocab_dict.json',
        cfg: DictConfig = None,
    ):
        """Creates vocab file from tokenizer if vocab file is None.

        Args:
            vocab_file_config_path: path to the vocab_file in the config
            vocab_dict_config_path: path to the vocab_dict in the config
            cfg: tokenizer config
        """
        if self.tokenizer is None:
            raise ValueError('Instantiate self.tokenizer before registering vocab from it.')
        else:
            if isinstance(self.tokenizer, AutoTokenizer):
                # extract vocab from tokenizer
                vocab_dict = self.tokenizer.tokenizer.get_vocab()

                # for fast and slow tokenizer vocabularies compatibility
                vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))

                # get hash of vocab_dict to create a unique directory to write vocab_dict and vocab_file
                m = hashlib.md5()
                if 'tokenizer_name' in cfg:
                    if cfg.tokenizer_name is not None:
                        # different pretrained models with the same vocab will have different hash
                        m.update(cfg.tokenizer_name.encode())
                # get string representation of vocab_dict
                vocab_dict_str = json.dumps(vocab_dict, sort_keys=True).encode()
                m.update(vocab_dict_str)
                vocab_dict_hash = m.hexdigest()

                hash_path = os.path.join(NEMO_NLP_TMP, vocab_dict_hash)
                os.makedirs(hash_path, exist_ok=True)

                vocab_json_src = os.path.join(hash_path, vocab_dict_config_path)

                with open(vocab_json_src, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(vocab_dict, indent=2, sort_keys=True) + '\n')
                self.register_artifact(config_path=vocab_dict_config_path, src=vocab_json_src)
                # create vocab file
                vocab_file_src = os.path.join(hash_path, vocab_file_config_path)
                with open(vocab_file_src, 'w', encoding='utf-8') as f:
                    for key in vocab_dict:
                        f.write(key + '\n')

                cfg.vocab_file = vocab_file_config_path
                self.register_artifact(config_path=vocab_file_config_path, src=vocab_file_src)
            else:
                logging.info(
                    f'Registering tokenizer vocab for {self.tokenizer} is not yet supported. Please override this method if needed.'
                )

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Override for LightningModule DDP initialization.
            Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_gpus
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            mpu.initialize_model_parallel(app_state.model_parallel_size)
            app_state.model_parallel_group = mpu.get_model_parallel_group()
            app_state.data_parallel_group = mpu.get_data_parallel_group()
            app_state.model_parallel_rank = torch.distributed.get_rank(group=app_state.model_parallel_group)
            app_state.data_parallel_rank = torch.distributed.get_rank(group=app_state.data_parallel_group)
            logging.info(f'mp_rank: {app_state.model_parallel_rank}')
            logging.info(f'dp_rank: {app_state.data_parallel_rank}')

    def configure_ddp(self, model: LightningModule, device_ids: List[int]) -> DistributedDataParallel:
        """ Override LightningModule ddp if using model parallel.

        Args:
            model (LightningModule): the LightningModule currently being optimized
            device_ids (List[int]): the list of GPU ids.

        Returns:
            DistributedDataParallel: DDP wrapped model
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info("Configuring DDP for model parallelism.")
            logging.info(f"data_parallel_group: {app_state.data_parallel_group}")
            # with model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial

            model = LightningDistributedDataParallel(
                model, device_ids, output_device=device_ids[0], process_group=app_state.data_parallel_group
            )
            return model

        else:
            logging.info("Did not detect model parallel using LightningModule.configure_ddp")
            return LightningModule.configure_ddp(self, model, device_ids)

    def setup(self, stage: str) -> None:
        """ PTL hook that is called after DDP is initialized.
            Called at the beginning of fit and test. 

        Args:
            stage (str): either 'fit' or 'test'
        """
        # TODO: implement model parallel for test stage
        if stage == 'fit':
            # adds self.bert_model config to .nemo file
            self.register_bert_model()

            app_state = AppState()

            if app_state.model_parallel_size is not None:

                if app_state.model_parallel_group is None:
                    self.init_model_parallel(app_state.global_rank, app_state.world_size)

                # Update PTL trainer to use our configure_ddp
                self._trainer.accelerator_backend.ddp_plugin.configure_ddp = self.configure_ddp

                if isinstance(self.bert_model, MegatronBertEncoder):
                    logging.info(f"restoring model parallel checkpoint: {self.bert_model._restore_path}")
                    # model parallel checkpoints need to be restored after torch.distributed is initialized
                    self.bert_model.restore_weights(self.bert_model._restore_path)

                    logging.info("replacing sampler with model parallel sampler")
                    mp_sampler = torch.utils.data.distributed.DistributedSampler(
                        self._train_dl.dataset,
                        num_replicas=app_state.data_parallel_size,
                        rank=app_state.data_parallel_rank,
                    )
                    mp_dl = self._trainer.replace_sampler(self._train_dl, mp_sampler)
                    self._train_dl = mp_dl
                else:
                    raise NotImplementedError(
                        f'The BERT encoder: {self.bert_model} does not support model parallelism yet.'
                    )
