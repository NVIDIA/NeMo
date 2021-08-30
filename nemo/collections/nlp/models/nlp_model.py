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

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from typing import Any, Dict, Optional, Union

import torch
from megatron import mpu
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
from nemo.collections.nlp.parts.nlp_overrides import NLPCheckpointConnector
from nemo.core.classes import ModelPT
from nemo.core.classes.exportable import Exportable
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import AppState, logging
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

    def _clip_gradients(self, optimizer, clip_val=None):
        """ Override of PTL Gradient Clipping.
            Enables model parallel gradient clipping from Megatron-LM.

        Args:
            optimizer ([type]): [description]
            clip_val ([type], optional): [description]. Defaults to None.
        """
        app_state = AppState()

        # get clip_val from trainer if None is provided
        if clip_val is None:
            clip_val = float(self._trainer.gradient_clip_val)

        if app_state.model_parallel_size is not None:
            model = self._trainer.get_model()
            parameters = model.parameters()
            if mpu.model_parallel_is_initialized():
                mpu.grads.clip_grad_norm(parameters=parameters, max_norm=clip_val)
            else:
                raise ValueError('Model parallel groups must be intialized to use model parallel gradient clipping.')

        else:
            return Accelerator._clip_gradients(self, optimizer, clip_val)

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

    # no rank check as model parallel models need to be saved on data parallel rank 0
    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into .nemo file
         You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_weights.ckpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """
        save_path = os.path.abspath(save_path)
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            self._default_save_to(save_path)
        else:
            # super.save_to only runs on global rank 0
            return super().save_to(save_path)

    def _default_save_to(self, save_path: str):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # each model parallel rank creates a .nemo file
            # after all .nemo files are created, each rank
            # will add their checkpoint to global rank 0

            base_dir = os.path.dirname(save_path)  # use the directory to merge mp_rank .nemo files into one

            # update save_path based on model parallel_rank
            base_path = os.path.splitext(save_path)[0]  # everything except the extension

            mp_save_path = f'{base_path}_mp_rank_{app_state.model_parallel_rank:02d}.nemo'

            if app_state.data_parallel_rank == 0:
                super()._default_save_to(mp_save_path)

            # barrier so that all processes have finished writing their weights before creating .nemo file
            torch.distributed.barrier()

            if is_global_rank_zero():
                # extract all tar files
                for mp_rank in range(app_state.model_parallel_size):
                    mp_tar_path = f'{base_path}_mp_rank_{mp_rank:02d}.nemo'
                    mp_tar = tarfile.open(mp_tar_path, 'r:gz')
                    mp_tar.extractall(path=os.path.join(base_dir, f'mp_rank_{mp_rank:02d}'))
                    mp_tar.close()
                    os.remove(mp_tar_path)

                # move rank 0 .nemo extract to base_path
                shutil.move(os.path.join(base_dir, 'mp_rank_00'), base_path)

                # move mp_rank_00 checkpoint to mp_rank_00 directory inside base_path
                os.mkdir(os.path.join(base_path, 'mp_rank_00'))
                shutil.move(os.path.join(base_path, 'model_weights.ckpt'), os.path.join(base_path, 'mp_rank_00'))

                # move other mp_rank checkpoints from base_dir to base_path
                for mp_rank in range(1, app_state.model_parallel_size):
                    os.mkdir(os.path.join(base_path, f'mp_rank_{mp_rank:02d}'))
                    shutil.move(
                        os.path.join(base_dir, f'mp_rank_{mp_rank:02d}', 'model_weights.ckpt'),
                        os.path.join(base_path, f'mp_rank_{mp_rank:02d}'),
                    )
                    # clean up leftover directory
                    shutil.rmtree(os.path.join(base_dir, f'mp_rank_{mp_rank:02d}'))

                # create tar file from base_path
                self._make_nemo_file_from_folder(save_path, base_path)

                # clean up base_path
                shutil.rmtree(base_path)

        elif is_global_rank_zero():
            return super()._default_save_to(save_path)
        else:
            return

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
        save_restore_connector: SaveRestoreConnector = None,
    ):
        """
        Restores model instance (weights and configuration) from .nemo file.

        Args:
            restore_path: path to .nemo file from which model should be instantiated
            override_config_path: path to a yaml config that will override the internal
                config file or an OmegaConf / DictConfig object representing the model config.
            map_location: Optional torch.device() to map the instantiated model to a device.
                By default (None), it will select a GPU if available, falling back to CPU otherwise.
            strict: Passed to load_state_dict. Set to True by default.
            return_config: If set to true, will return just the underlying config of the restored
                model as an OmegaConf DictConfig object without instantiating the model.
            trainer: PyTorch Lightning trainer. Must be passed in order to use model parallel .nemo

            Example:
                ```
                model = nemo.collections.nlp.models.TokenClassificationModel.restore_from('token_classification.nemo')
                assert isinstance(model, nemo.collections.nlp.models.TokenClassificationModel)
                ```

        Returns:
            An instance of type cls or its underlying config (if return_config is set).
        """
        if save_restore_connector is None:
            save_restore_connector = SaveRestoreConnector()

        if not os.path.exists(restore_path):
            raise FileNotFoundError(f"Can't find {restore_path}")

        app_state = AppState()
        app_state.model_restore_path = os.path.abspath(os.path.expanduser(restore_path))

        # detect if we have a model parallel .nemo file
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            # detect if model parallel from tarfile
            tar = tarfile.open(app_state.model_restore_path, "r:gz")
            names = tar.getnames()
            mp_ranks = []
            for name in names:
                if 'mp_rank' in name:
                    mp_ranks.append(name)
            if mp_ranks:
                app_state.model_parallel_size = len(mp_ranks) // 2  # directory and file are included in getnames()

                # get checkpoint version
                checkpoint_version_member = None
                for member in tar.getmembers():
                    if 'megatron_checkpoint_version.json' in member.name:
                        checkpoint_version_member = member
                tar.extract(checkpoint_version_member, tmpdir)
                with open(checkpoint_version_member.name, 'r') as f:
                    checkpoint_version = json.load(f).get('checkpoint_version', None)
                logging.info(
                    (
                        f'Detected model parallel .nemo file: {restore_path}. '
                        f'Assuming megatron model parallelism with '
                        f'model_parallel_size: {app_state.model_parallel_size} '
                        f'and checkpoint version: {checkpoint_version}'
                    )
                )
            tar.close()
            os.chdir(cwd)

        if app_state.model_parallel_size is not None:
            if not isinstance(trainer, Trainer):
                raise ValueError("trainer must be a PyTorch Lightning Trainer to restore model parallel .nemo files.")

            if checkpoint_version is None:
                raise ValueError(
                    "Restoring from megatron model parallel .nemo but could not find megatron checkpoint version."
                )
            else:
                logging.info(f"Setting megatron checkpoint version: {checkpoint_version}")
                set_checkpoint_version(checkpoint_version)

            app_state.world_size = trainer.num_gpus * trainer.num_nodes

            if trainer.local_rank is not None:
                app_state.local_rank = trainer.local_rank
            else:
                raise ValueError("trainer.local_rank is None. local_rank needed to restore model parallel models.")

            model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)
            app_state.model_parallel_rank = model_parallel_rank

            cls.update_save_restore_connector(save_restore_connector)
            restored_model = cls._save_restore_connector.restore_from(
                cls, app_state.model_restore_path, override_config_path, map_location, strict, return_config
            )
            restored_model.set_trainer(trainer)
            return restored_model
        else:
            return super().restore_from(
                app_state.model_restore_path,
                override_config_path,
                map_location,
                strict,
                return_config,
                save_restore_connector=save_restore_connector,
            )

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
