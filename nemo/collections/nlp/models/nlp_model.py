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

import glob
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from typing import Any, Dict, Optional, Union

import torch
from apex import mpu
from megatron.checkpointing import get_checkpoint_version, set_checkpoint_version
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities import rank_zero_only
from transformers import TRANSFORMERS_CACHE

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.modules import BertModule, MegatronBertEncoder
from nemo.collections.nlp.modules.common.megatron.megatron_encoder import MegatronEncoderModule
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPCheckpointConnector, NLPSaveRestoreConnector
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.exp_manager import configure_checkpointing
from nemo.utils.get_rank import is_global_rank_zero

__all__ = ['NLPModel']

NEMO_NLP_TMP = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), "nemo_nlp_tmp")

os.makedirs(NEMO_NLP_TMP, exist_ok=True)


class NLPModel(ModelPT, Exportable):
    """Base class for NLP Models.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        self.set_world_size(trainer)

    def register_artifact(
        self, config_path: str, src: str, verify_src_exists: bool = False,
    ):
        """ Overrides ModelPT register_artifact default behavior. NLP models usually need artifacts that are optional."""
        return super().register_artifact(config_path, src, verify_src_exists=verify_src_exists)

    @rank_zero_only
    def register_bert_model(self):
        """Adds encoder config to .nemo archive for Jarvis.
        """
        # check if there is an encoder, warn if not
        if self.bert_model is None:
            raise ValueError('Instantiate self.bert_model before registering it.')
        else:
            # get encoder config and create source for artifact
            if isinstance(self.bert_model, MegatronBertEncoder):
                pretrained_model_name = self.bert_model._model_name
                encoder_config_path = pretrained_model_name + '_encoder_config'
                encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path + '.json')
                config_for_json = OmegaConf.to_container(self.bert_model.config)
                with open(encoder_config_src, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(config_for_json, indent=2, sort_keys=True) + '\n')
                self.register_artifact('language_model.config_file', encoder_config_src)  # for .nemo
            elif isinstance(self.bert_model, BertModule):
                # HuggingFace Transformer Config
                pretrained_model_name = self.bert_model.name_or_path
                # Some HF names have "/" in them so we replace with _
                pretrained_model_name = pretrained_model_name.replace("/", "_")
                encoder_config_path = pretrained_model_name + '_encoder_config'
                encoder_config_src = os.path.join(NEMO_NLP_TMP, encoder_config_path + '.json')
                self.bert_model.config.to_json_file(encoder_config_src)  # name requested by jarvis team
                self.register_artifact('language_model.config_file', encoder_config_src)  # for .nemo
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
        vocab_file = None
        if cfg.vocab_file:
            vocab_file = self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file)
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
        vocab_dict_config_path: str = 'tokenizer_vocab_dict',
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

                cfg.vocab_file = vocab_file_src
                self.register_artifact(config_path=vocab_file_config_path, src=vocab_file_src)
            else:
                logging.info(
                    f'Registering tokenizer vocab for {self.tokenizer} is not yet supported. Please override this method if needed.'
                )

    def setup(self, stage: str) -> None:
        """ PTL hook that is called on all DDP processes. """

        if stage == 'fit':

            # adds self.bert_model config to .nemo file
            if hasattr(self, 'bert_model') and self.bert_model is not None:
                self.register_bert_model()

            app_state = AppState()

            if app_state.model_parallel_size is not None:

                self._trainer.checkpoint_connector = NLPCheckpointConnector(self._trainer)

                # # Configure checkpointing for model parallel
                # if app_state.create_checkpoint_callback:
                #     # global rank 0 is configured by exp_manager
                #     if not is_global_rank_zero() and app_state.data_parallel_rank == 0:
                #         configure_checkpointing(
                #             self._trainer,
                #             app_state.log_dir,
                #             app_state.checkpoint_name,
                #             app_state.checkpoint_callback_params,
                #         )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ LightningModule hook that's used to save things in addition to model weights. """

        if hasattr(self, "bert_model") and isinstance(self.bert_model, MegatronBertEncoder):
            checkpoint['checkpoint_version'] = get_checkpoint_version()
        return None

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ LightningModule hook that's used to restore things saved with on_save_checkpoint."""

        if hasattr(self, "bert_model") and isinstance(self.bert_model, MegatronBertEncoder):
            if get_checkpoint_version():
                assert (
                    checkpoint['checkpoint_version'] == get_checkpoint_version()
                ), 'checkpoint version found on_load_checkpoint different than get_checkpoint_version'
            else:
                set_checkpoint_version(checkpoint['checkpoint_version'])
                logging.info(f"Setting Megatron checkpoint version: {checkpoint['checkpoint_version']}")
        return None

    @rank_zero_only
    def register_megatron_checkpoint_version(self):
        """ Adds checkpoint version to .nemo archive """
        if self.has_megatron_encoder:
            checkpoint_version = get_checkpoint_version()
            if checkpoint_version is None:
                raise ValueError('Unable to get megatron checkpoint version.')
            else:
                checkpoint_version_dict = {'checkpoint_version': checkpoint_version}
                checkpoint_version_path = 'megatron_checkpoint_version.json'
                checkpoint_version_src = os.path.join(NEMO_NLP_TMP, checkpoint_version_path)
                with open(checkpoint_version_src, 'w') as f:
                    f.write(json.dumps(checkpoint_version_dict))
                self.register_artifact(checkpoint_version_path, checkpoint_version_src)
        else:
            raise ValueError('Registering Megatron checkpoint version but no Megatron encoder detected.')

    @staticmethod
    def _unpack_nemo_file(path2file: str, out_folder: str) -> str:
        return super(NLPModel, NLPModel)._unpack_nemo_file(path2file, out_folder)

    @staticmethod
    def _make_nemo_file_from_folder(filename, source_dir):
        return super(NLPModel, NLPModel)._make_nemo_file_from_folder(filename, source_dir)

    @property
    def input_module(self):
        return self.bert_model

    @property
    def output_module(self):
        return self.classifier

    @property
    def has_megatron_encoder(self):
        if hasattr(self, 'bert_model'):
            if isinstance(self.bert_model, MegatronBertEncoder):
                return True
            else:
                return False
        elif hasattr(self, 'encoder'):
            if isinstance(self.encoder, MegatronEncoderModule):
                return True
            else:
                return False
        else:
            return False

    @property
    def is_model_parallel_initialized(self):
        app_state = AppState()
        if app_state.model_parallel_group is not None:
            return True
        else:
            return False

    def restore_megatron_encoder_weights(self):
        """ Model parallel weights need to be restored after DDP is initialized and 
            model parallel ranks are known.
        """
        if hasattr(self, 'bert_model'):
            if isinstance(self.bert_model, MegatronBertEncoder):
                logging.info(f"Restoring from pretrained model parallel checkpoint: {self.bert_model._restore_path}")
                self.bert_model.restore_weights(self.bert_model._restore_path)
        elif hasattr(self, 'encoder'):
            if isinstance(self.encoder, MegatronEncoderModule):
                logging.info(f"Restoring from pretrained model parallel checkpoint: {self.encoder.checkpoint_file}")
                self.encoder._encoder.restore_weights(self.encoder.checkpoint_file)
