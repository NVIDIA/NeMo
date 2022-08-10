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

import itertools
import json
import random
from multiprocessing import Value
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as pt_data
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu

from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.losses import NLLLoss
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTRetrievalModelConfig
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.modules.common.transformer import AttentionBridge, TopKSequenceGenerator
from nemo.collections.nlp.data import RetrievalTranslationDataset, TarredTranslationDataset, TranslationDataset
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin, NLPSaveRestoreConnector
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils, timers

__all__ = ['MTRetrievalModel']

class MTRetrievalModel(MTEncDecModel):
    """
    Proposed Retrieval-Model for NMT by conditioning on Perceiver embeddings
    Things to implement
    > See what methods to override and nothing more. Maybe make a new encoder class as well?
    > Add Config file for this?
    > load perceiver model
    > compute perceiver embeddings
    > condition on perceiver embeddings during train/test/val
    > Not to add positional embeddings

    Second step
    > Implement cross attention styel
    > Do monolingual stuff
    """

    def __init__(self, cfg: MTRetrievalModelConfig, trainer: Trainer = None):
        # when to call this? is this right time?
        super().__init__(cfg=cfg, trainer=trainer)

        # TODO: load configs params
        

        # load the perceiver
        self.retrieval_encoder = MegatronBARTModel.restore_from(
                    cfg.get("retrieval_encoder"),
                    trainer=trainer, 
                    save_restore_connector=NLPSaveRestoreConnector(),
                )

    def encode_neighbor(self, padded_batch):
        '''
        Logic for computing latent tokens from perceiver embeddings
        padded_batch: List of padded sentences to encode
        '''
        batch_size = len(padded_batch)
        # 64 * 1024 etc
        hidden_dim = self.retrieval_encoder.cfg.hidden_size * self.retrieval_encoder.cfg.hidden_steps
        latents = np.zeros((batch_size, hidden_dim)).astype(np.float16)
        padded_batch = torch.LongTensor(padded_batch).cuda()
        mask = padded_batch != self.retrieval_encoder.tokenizer.pad_id
        latent = self.retrieval_encoder.encode(padded_batch, mask)
        latent = latent.contiguous().view(len(padded_batch), -1).data.cpu().numpy().astype(np.float16)
        return latents

    @typecheck()
    def forward(self, src, src_mask, tgt, tgt_mask, nn_src_ids, nn_tgt_ids):
        if self.validate_input_ids:
            # test src/tgt for id range (i.e., hellp in catching wrong tokenizer)
            # TODO: Add for nn_src_ids and nn_tgt_ids as well
            self.test_encoder_ids(src, raise_error=True)
            self.test_decoder_ids(tgt, raise_error=True)
        
        # Encode neighbors using retrieval_encoder
        for 

        src_hiddens = self.encoder(input_ids=src, encoder_mask=src_mask)
        tgt_hiddens = self.decoder(
            input_ids=tgt, decoder_mask=tgt_mask, encoder_embeddings=src_hiddens, encoder_mask=src_mask
        )
        log_probs = self.log_softmax(hidden_states=tgt_hiddens)
        return log_probs

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)
        src_ids, src_mask, tgt_ids, tgt_mask, labels, nn_src_ids, nn_tgt_ids = batch
        log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask, nn_src_ids, nn_tgt_ids)
        train_loss = self.loss_fn(log_probs=log_probs, labels=labels)
        tensorboard_logs = {
            'train_loss': train_loss,
            'lr': self._optimizer.param_groups[0]['lr'],
        }

        return {'loss': train_loss, 'log': tensorboard_logs}

    def eval_step(self, batch, batch_idx, mode, dataloader_idx=0):
        for i in range(len(batch)):
            if batch[i].ndim == 3:
                # Dataset returns already batched data and the first dimension of size 1 added by DataLoader
                # is excess.
                batch[i] = batch[i].squeeze(dim=0)

        if self.multilingual:
            self.source_processor = self.source_processor_list[dataloader_idx]
            self.target_processor = self.target_processor_list[dataloader_idx]

        src_ids, src_mask, tgt_ids, tgt_mask, labels, nn_src_ids, nn_tgt_ids = batch
        log_probs = self(src_ids, src_mask, tgt_ids, tgt_mask, nn_src_ids, nn_tgt_ids)
        eval_loss = self.eval_loss_fn(log_probs=log_probs, labels=labels)
        # this will run encoder twice -- TODO: potentially fix
        inputs, translations = self.batch_translate(src=src_ids, src_mask=src_mask)
        if dataloader_idx == 0:
            getattr(self, f'{mode}_loss')(loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1])
        else:
            getattr(self, f'{mode}_loss_{dataloader_idx}')(
                loss=eval_loss, num_measurements=log_probs.shape[0] * log_probs.shape[1]
            )
        np_tgt = tgt_ids.detach().cpu().numpy()
        ground_truths = [self.decoder_tokenizer.ids_to_text(tgt) for tgt in np_tgt]
        ground_truths = [self.target_processor.detokenize(tgt.split(' ')) for tgt in ground_truths]
        num_non_pad_tokens = np.not_equal(np_tgt, self.decoder_tokenizer.pad_id).sum().item()
        return {
            'inputs': inputs,
            'translations': translations,
            'ground_truths': ground_truths,
            'num_non_pad_tokens': num_non_pad_tokens,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, batch_idx, 'test', dataloader_idx)

    @classmethod
    def _setup_dataset_from_config(
        cls,
        cfg: DictConfig,
        encoder_tokenizer,
        decoder_tokenizer,
        global_rank,
        world_size,
        multilingual,
        multilingual_ids,
    ):
        if cfg.get("use_tarred_dataset", False) or cfg.get("dataset_type", "") == "tarred":
            # TODO: See if retrieval models can support tarred_datasets
            if cfg.get("metadata_file") is None:
                raise FileNotFoundError("Trying to use tarred data set but could not find metadata path in config.")
            metadata_file_list = cfg.get('metadata_file')
            tar_files_list = cfg.get('tar_files', None)
            if isinstance(metadata_file_list, str):
                metadata_file_list = [metadata_file_list]
            if tar_files_list is not None and isinstance(tar_files_list, str):
                tar_files_list = [tar_files_list]
            if tar_files_list is not None and len(tar_files_list) != len(metadata_file_list):
                raise ValueError('The config must have the same number of tarfile paths and metadata file paths.')

            datasets = []
            for idx, metadata_file in enumerate(metadata_file_list):
                with open(metadata_file) as metadata_reader: 
                    metadata = json.load(metadata_reader)
                if tar_files_list is None:
                    tar_files = metadata.get('tar_files')
                    if tar_files is not None:
                        # update absolute path of tar files based on metadata_file path
                        valid_tar_files = []
                        metadata_basedir = os.path.abspath(os.path.dirname(metadata_file))
                        updated_fn = 0
                        for fn in tar_files:
                            # if a file does not exist, look in metadata file directory
                            if os.path.exists(fn):
                                valid_fn = fn
                            else:
                                updated_fn += 1
                                valid_fn = os.path.join(metadata_basedir, os.path.basename(fn))
                                if not os.path.exists(valid_fn):
                                    raise RuntimeError(
                                        f"File in tarred dataset is missing from absolute and relative paths {fn}"
                                    )

                            valid_tar_files.append(valid_fn)

                        tar_files = valid_tar_files

                        logging.info(f'Updated the path of {updated_fn} tarred files')
                        logging.info(f'Loading from tarred dataset {tar_files}')
                else:
                    tar_files = tar_files_list[idx]
                    if metadata.get('tar_files') is not None:
                        logging.info(
                            f'Tar file paths found in both cfg and metadata using one in cfg by default - {tar_files}'
                        )

                dataset = TarredTranslationDataset(
                    text_tar_filepaths=tar_files,
                    metadata_path=metadata_file,
                    encoder_tokenizer=encoder_tokenizer,
                    decoder_tokenizer=decoder_tokenizer,
                    shuffle_n=cfg.get("tar_shuffle_n", 100),
                    shard_strategy=cfg.get("shard_strategy", "scatter"),
                    global_rank=global_rank,
                    world_size=world_size,
                    reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                    prepend_id=multilingual_ids[idx] if multilingual else None,
                )
                datasets.append(dataset)

            if len(datasets) > 1:
                dataset = ConcatDataset(
                    datasets=datasets,
                    sampling_technique=cfg.get('concat_sampling_technique'),
                    sampling_temperature=cfg.get('concat_sampling_temperature'),
                    sampling_probabilities=cfg.get('concat_sampling_probabilities'),
                    global_rank=global_rank,
                    world_size=world_size,
                )
            else:
                dataset = datasets[0]
        else:
            src_file_list = cfg.src_file_name
            tgt_file_list = cfg.tgt_file_name
            retrieval_indices_file_list = cfg.retrieval_indices
            if isinstance(src_file_list, str):
                src_file_list = [src_file_list]
            if isinstance(tgt_file_list, str):
                tgt_file_list = [tgt_file_list]
            if isinstance(retrieval_indices_file_list, str):
                retrieval_indices_file_list = [retrieval_indices_file_list]
            if len(src_file_list) != len(tgt_file_list):
                raise ValueError('The same number of filepaths must be passed in for source and target.')
            if len(src_file_list) != len(retrieval_indices_file_list):
                raise ValueError('The same number of filepaths must be passed in for src/tgt and retrieval indices.')

            datasets = []
            for idx, src_file in enumerate(src_file_list):
                dataset = RetrievalTranslationDataset(
                    dataset_src=str(Path(src_file).expanduser()),
                    dataset_tgt=str(Path(tgt_file_list[idx]).expanduser()),
                    retrieval_indices=str(Path(retrieval_indices_file_list[idx]).expanduser()),
                    retrieval_db_src=str(Path(cfg.get('retrieval_db_src', None)).expanduser()),
                    retrieval_db_tgt=str(Path(cfg.get('retrieval_db_tgt', None)).expanduser()),
                    tokens_in_batch=cfg.tokens_in_batch,
                    clean=cfg.get("clean", False),
                    max_seq_length=cfg.get("max_seq_length", 512),
                    min_seq_length=cfg.get("min_seq_length", 1),
                    max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                    max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                    cache_ids=cfg.get("cache_ids", False),
                    cache_data_per_node=cfg.get("cache_data_per_node", False),
                    use_cache=cfg.get("use_cache", False),
                    reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                    prepend_id=multilingual_ids[idx] if multilingual else None,
                    retrieval_nns=cfg.get("retrieval_nns", 1),
                )
                dataset.batchify(encoder_tokenizer, decoder_tokenizer)
                datasets.append(dataset)
            if len(datasets) > 1:
                dataset = ConcatDataset(
                    datasets=datasets,
                    shuffle=cfg.get('shuffle'),
                    sampling_technique=cfg.get('concat_sampling_technique'),
                    sampling_temperature=cfg.get('concat_sampling_temperature'),
                    sampling_probabilities=cfg.get('concat_sampling_probabilities'),
                    global_rank=global_rank,
                    world_size=world_size,
                )
            else:
                dataset = datasets[0]

        return dataset

    @classmethod
    def _setup_eval_dataset_from_config(
        cls, cfg: DictConfig, multilingual: bool, multilingual_ids, encoder_tokenizer, decoder_tokenizer
    ):
        src_file_name = cfg.get('src_file_name')
        tgt_file_name = cfg.get('tgt_file_name')
        retrieval_indices = cfg.get('retrieval_indices')
        retrieval_db_src = cfg.get('retrieval_db_src')
        retrieval_db_tgt = cfg.get('retrieval_db_tgt')

        # Some checks
        if src_file_name is None or tgt_file_name is None:
            raise ValueError(
                'Validation dataloader needs both cfg.src_file_name and cfg.tgt_file_name to not be None.'
            )
        elif retrieval_indices is None or retrieval_db_src is None or retrieval_db_tgt is None:
            raise ValueError(
                'Validation dataloader needs cfg.retrieval_indices, cfg.retrieval_db_src, cfg.retrieval_db_tgt \
                to not be None for retrieval mode.'
            )
        else:
            # convert src_file_name and tgt_file_name to list of strings
            if isinstance(src_file_name, str):
                src_file_list = [src_file_name]
            elif isinstance(src_file_name, ListConfig):
                src_file_list = src_file_name
            else:
                raise ValueError("cfg.src_file_name must be string or list of strings")
            if isinstance(tgt_file_name, str):
                tgt_file_list = [tgt_file_name]
            elif isinstance(tgt_file_name, ListConfig):
                tgt_file_list = tgt_file_name
            else:
                raise ValueError("cfg.tgt_file_name must be string or list of strings")
            if isinstance(retrieval_indices, str):
                retrieval_indices_file_list = [retrieval_indices]
            elif isinstance(retrieval_indices, ListConfig):
                retrieval_indices_file_list = retrieval_indices
            else:
                raise ValueError("cfg.retrieval_indices must be string or list of strings")
        if len(src_file_list) != len(tgt_file_list):
            raise ValueError('The same number of filepaths must be passed in for source and target validation.')

        if len(src_file_list) != len(retrieval_indices_file_list):
            raise ValueError('The same number of filepaths must be passed in for source and retrieval validation.')

        datasets = []
        prepend_idx = 0
        for idx, src_file in enumerate(src_file_list):
            if multilingual:
                # TODO: Remove multilingual references?
                prepend_idx = idx
            dataset = RetrievalTranslationDataset(
                dataset_src=str(Path(src_file).expanduser()),
                dataset_tgt=str(Path(tgt_file_list[idx]).expanduser()),
                retrieval_indices=str(Path(retrieval_indices_file_list[idx]).expanduser()),
                retrieval_db_src=str(Path(cfg.get('retrieval_db_src', None)).expanduser()),
                retrieval_db_tgt=str(Path(cfg.get('retrieval_db_tgt', None)).expanduser()),
                tokens_in_batch=cfg.tokens_in_batch,
                clean=cfg.get("clean", False),
                max_seq_length=cfg.get("max_seq_length", 512),
                min_seq_length=cfg.get("min_seq_length", 1),
                max_seq_length_diff=cfg.get("max_seq_length_diff", 512),
                max_seq_length_ratio=cfg.get("max_seq_length_ratio", 512),
                cache_ids=cfg.get("cache_ids", False),
                cache_data_per_node=cfg.get("cache_data_per_node", False),
                use_cache=cfg.get("use_cache", False),
                reverse_lang_direction=cfg.get("reverse_lang_direction", False),
                prepend_id=multilingual_ids[prepend_idx] if multilingual else None,
                retrieval_nns=cfg.get("retrieval_nns", 1),
            )
            dataset.batchify(encoder_tokenizer, decoder_tokenizer)
            datasets.append(dataset)
        return datasets

# TODO Add batch_translate and other inference time logic