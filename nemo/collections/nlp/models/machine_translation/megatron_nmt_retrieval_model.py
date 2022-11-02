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
from torch.utils.data import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from sacrebleu import corpus_bleu

from nemo.collections.common.data import ConcatDataset
from nemo.collections.common.losses import NLLLoss
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTRetrievalModelConfig
from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.modules.common.transformer import AttentionBridge, TopKSequenceGenerator
from nemo.collections.nlp.data import RetrievalTranslationDataset, TarredTranslationDataset, TranslationDataset
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector, NLPDDPStrategy
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils, timers
from nemo.utils.app_state import AppState
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel

from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)

try:
    from apex.transformer.enums import ModelType
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.common import build_model
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.utils import (
        get_num_microbatches,
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    ModelType = ApexGuardDefaults()
    HAVE_APEX = False

__all__ = ['MegatronNMTRetrievalModel']

class MegatronNMTRetrievalModel(MegatronNMTModel):
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

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        # when to call this? is this right time?
        super().__init__(cfg=cfg, trainer=trainer)
        
        # TODO: Update from config
        self.num_neighbors = 1
        # TODO: load configs params
        # load the perceiver
        # TODO: See if this is correct? 
        # TODO: Do I need to call apex's build_model here?
        # TODO: Need to extract encoder from MegatronBARTModel which is perceiver encoder and transformer decoder
        if self.cfg.retriever.type == 'perceiver':
            # app_state = AppState()
            # if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            #     app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            #     (
            #         app_state.tensor_model_parallel_rank,
            #         app_state.pipeline_model_parallel_rank,
            #         app_state.model_parallel_size,
            #         app_state.data_parallel_size,
            #         app_state.pipeline_model_parallel_split_rank,
            #         app_state.virtual_pipeline_model_parallel_rank
            #     ) = fake_initialize_model_parallel(
            #         world_size=app_state.model_parallel_size,
            #         rank=trainer.global_rank,
            #         tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
            #         pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
            #         pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            #     )
            # if parallel_state.is_unitialized():
            #     def placeholder():
            #         return
            #     if trainer.strategy.launcher is not None:
            #         trainer.strategy.launcher.launch(placeholder, trainer=trainer)
            #     trainer.strategy.setup_environment()

            # Need to reconfigure micro batch calculator with apex for new p-tuning session

            pretrained_cfg = MegatronBARTModel.restore_from(cfg.retriever.get("encoder_path"), trainer=trainer, return_config=True)
            # app_state = AppState()
            # _reconfigure_microbatch_calculator(
            #     rank=app_state.global_rank,
            #     rampup_batch_size=None,
            #     global_batch_size=pretrained_cfg.global_batch_size,
            #     micro_batch_size=pretrained_cfg.micro_batch_size,
            #     data_parallel_size=parallel_state.get_data_parallel_world_size(),
            # )

            OmegaConf.set_struct(pretrained_cfg, True)
            with open_dict(pretrained_cfg):
                pretrained_cfg.masked_softmax_fusion = False
                pretrained_cfg.global_batch_size = cfg.global_batch_size
                pretrained_cfg.micro_batch_size = cfg.micro_batch_size
                
            self.retrieval_encoder = MegatronBARTModel.restore_from(
                        cfg.retriever.get("encoder_path"),
                        trainer=trainer, 
                        override_config_path=pretrained_cfg,
                        save_restore_connector=NLPSaveRestoreConnector(),
                    )
            self.retrieval_encoder.freeze()
        else:
            self.retrieval_encoder = None
        # TODO See how to get retrieval tokenizer from the model


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

    def build_train_valid_test_datasets(self):
        """Builds the train, validation, and test datasets."""
        self._train_without_neighbors_ds = self.build_memmap_dataset_from_config(self._cfg.train_ds)
        self._retrieval_ds = self.build_memmap_dataset_from_config(
            self._cfg.retrieval_ds,
            encoder_tokenizer=self.retrieval_encoder.tokenizer,
            decoder_tokenizer=self.retrieval_encoder.tokenizer,
            retrieval_dataset=True
            )
        self._train_ds = RetrievalSeq2Seq(
            self._train_without_neighbors_ds,
            self._retrieval_ds,
            self._cfg.train_ds.nn_mapping,
            self._cfg.train_ds.num_neighbors,
        )

        if self._cfg.validation_ds.get("dataset_type", "text") != "text":
            raise ValueError(f"Validation dataset type must be 'text', found {self._cfg.validation_ds.dataset_type}")

        # ! see if there are any modifications to be made here for diff b/w train and val
        # ! especially get_item or collate_fn could be different for em
        # return src_ids, src_mask, tgt_ids, tgt_mask, labels format vs dictionary in text_mmap
        self._validation_ds = MTEncDecModel._setup_eval_dataset_from_config(
            cfg=self._cfg.validation_ds,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
        )
        # TODO: Change this to self._validation_without_neighbors_ds
        # if isinstance(self._validation_without_neighbors_ds, List):
        #     self._validation_ds = []
        #     for idx, ds in enumerate(self._validation_without_neighbors_ds):
        #         self._validation_ds.append(RetrievalSeq2Seq(
        #             ds,
        #             self._retrieval_ds,
        #             self._cfg.validation_ds.nn_mapping[idx],
        #             self._cfg.validation_ds.num_neighbors,
        #         ))
        # else:
        #     self._validation_ds = RetrievalSeq2Seq(
        #         self._validation_without_neighbors_ds,
        #         self._retrieval_ds,
        #         self._cfg.validation_ds.nn_mapping,
        #         self._cfg.validation_ds.num_neighbors,
        #     )

        # Test data config is optional.
        if hasattr(self._cfg, 'test_ds'):
            if self._cfg.validation_ds.get("dataset_type", "text") != "text":
                raise ValueError(f"Test dataset type must be 'text', found {self._cfg.test_ds.dataset_type}")
            self._test_without_neighbors_ds = MTEncDecModel._setup_eval_dataset_from_config(
                cfg=self._cfg.validation_ds,
                multilingual=self.multilingual,
                multilingual_ids=self.multilingual_ids,
                encoder_tokenizer=self.encoder_tokenizer,
                decoder_tokenizer=self.decoder_tokenizer,
            )
            if isinstance(self._test_without_neighbors_ds, List):
                self._test_ds = []
                for idx,ds in enumerate(self._test_without_neighbors_ds):
                    self._test_ds.append(RetrievalSeq2Seq(
                        ds,
                        self._retrieval_ds,
                        self._cfg.test_ds.nn_mapping[idx],
                        self._cfg.test_ds.num_neighbors,
                    ))
            else:
                self._test_ds = RetrievalSeq2Seq(
                    self._test_without_neighbors_ds,
                    self._retrieval_ds,
                    self._cfg.test_ds.nn_mapping,
                    self._cfg.test_ds.num_neighbors,
                )
            

    def process_global_batch_for_text_translation_datasets(self, batch):
        #TODO: See how to modify this for retrievals. This is called in eval_step
        """Override parent process_batch since TranslationDataset does not return dictionaries."""
        # Convert each microbatch into a dictionary.
        src_ids, src_mask, tgt_ids, tgt_mask, labels = batch
        # ! for vanilla retrieval can just add src_ids and nn_ids here only to make it work
        batch = {
            'text_enc': src_ids,
            'text_dec': tgt_ids,
            'labels': labels,
            'enc_mask': src_mask.long(),  # super().process_batch() expects torch.int64
            'dec_mask': tgt_mask.long(),  # super().process_batch() expects torch.int64
            'loss_mask': tgt_mask.long(),  # super().process_batch() expects torch.int64
        }

        # Parent function will pad microbatches to the same length.
        return self._process_global_batch_without_megatron_batch_sampler([batch], tokenizer=self.encoder_tokenizer)
        
    def process_global_batch(self, global_batch):
        # ! modify for retrieval. This is called in train_step
        # ! See what the collate function of retrievalseq2seq returns
        return [
            global_batch["text_enc"],
            global_batch["text_dec"],
            global_batch["loss_mask"],
            global_batch["labels"],
            global_batch["enc_mask"],
            global_batch["dec_mask"],
            global_batch.get("nn_src_list", None),
            global_batch.get("nn_tgt_list", None),
            global_batch.get("nn_src_mask_list", None),
            global_batch.get("nn_tgt_mask_list", None),
        ]

    def get_forward_output_and_loss_func(self, val=False):
        """
            This function is used to create a closure that is passed to the apex fwd/bwd functions.
            Overrided from enc_dec_model.py
        """
        def fwd_output_and_loss_func_perceiver(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            (
                encoder_input_ids,
                decoder_input_ids, 
                loss_mask, 
                lm_labels, 
                encoder_attn_mask, 
                decoder_attn_mask, 
                nn_src_ids, 
                nn_tgt_ids,
                nn_src_mask,
                nn_tgt_mask, 
            ) = batch
            """
            1. Get the embeddings from the embedding matrix and add positional embeddings
            2. Get the perceiver encodings from self.retriever. Embedding dropout? 
            3. Concatenate the two
            4. Pass it to the model forward using encoder inputs option
            5. Figure out how to do batching here properly
            """

            # Compute perceiver embeddings. Dim is [B x S x H]
            latent_src = self.retrieval_encoder.encode(nn_src_ids, nn_src_mask)
            latent_tgt = self.retrieval_encoder.encode(nn_tgt_ids, nn_tgt_mask)

            if model.encoder_cfg.get("position_embedding_type", "learned_absolute") != 'relative':
                enc_position_ids = build_position_ids(encoder_input_ids)
            else:
                enc_position_ids = None

            enc_input = model.encoder_embedding(encoder_input_ids, enc_position_ids, token_type_ids=None)
            prepend_length = 2 * self.num_neighbors * latent_src.shape[1]
            mask_prepend = torch.ones((encoder_attn_mask.shape[0], prepend_length)).to(encoder_attn_mask.device)
            encoder_attn_mask = torch.cat((mask_prepend, encoder_attn_mask), dim=1)

            # join latents with the encoder input embeddings
            # Concatenate along the sequence dimension
            enc_input = torch.cat([latent_src, latent_tgt, enc_input.transpose(0,1)], dim=1)
            output = model(
                enc_input_ids=None,
                enc_attn_mask=encoder_attn_mask,
                dec_input_ids=decoder_input_ids,
                dec_attn_mask=decoder_attn_mask,
                token_type_ids=None,
                labels=lm_labels,
                enc_output=None,
                enc_output_attn_mask=None,
                enc_input=enc_input, 
                output_enc_hidden_only=False,
            )

            def loss_func(output_tensor):
                loss = self.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output, loss_func
        
        def fwd_output_and_loss_func_concat(batch, model):
            raise NotImplementedError
            batch = [x.cuda(non_blocking=True) for x in batch]
            encoder_input_ids, decoder_input_ids, loss_mask, lm_labels, encoder_attn_mask, decoder_attn_mask, nn_src_list_ids, nn_tgt_list = batch
            # ! for retrieval embeddings figure out how to do this
            """
            1. Get the embeddings from the embedding matrix and add positional embeddings
            2. Get the perceiver encodings from self.retriever. Embedding dropout? 
            3. Concatenate the two
            4. Pass it to the model forward using encoder inputs option
            5. Figure out how to do batching here properly
            """
            output = model(
                encoder_input_ids,  # enc_input_ids
                encoder_attn_mask,  # enc_attn_mask
                decoder_input_ids,  # dec_input_ids
                decoder_attn_mask,  # dec_attn_mask
                None,  # token_type_ids
                lm_labels,  # labels
            )

            def loss_func(output_tensor):
                loss = self.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output, loss_func

        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            encoder_input_ids, decoder_input_ids, loss_mask, lm_labels, encoder_attn_mask, decoder_attn_mask = batch
            # ! invoke model embedding lookup 
            # concentate 
            output = model(
                encoder_input_ids,  # enc_input_ids
                encoder_attn_mask,  # enc_attn_mask
                decoder_input_ids,  # dec_input_ids
                decoder_attn_mask,  # dec_attn_mask
                None,  # token_type_ids
                lm_labels,  # labels
            )

            def loss_func(output_tensor):
                loss = self.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output, loss_func
        
        if val is True:
            # TODO how to call super output and loss func insteaad
            return fwd_output_and_loss_func
        elif self.cfg.retriever.get('type', False) == 'text':
            return fwd_output_and_loss_func_concat
        elif self.cfg.retriever.get('type', False) == 'perceiver':
            return fwd_output_and_loss_func_perceiver
        else:
            raise NotImplementedError
            
class RetrievalSeq2Seq(Dataset):
    """
    This class defines dataset for retrieval Seq2Seq model."""

    def __init__(
        self,
        train_text_memmap_dataset,
        retrieval_text_memmap_dataset,
        nn_mapping,
        num_neighbors = 10,
    ):
        self.train_text_memmap_dataset = train_text_memmap_dataset
        self.retrieval_text_memmap_dataset = retrieval_text_memmap_dataset
        self.num_neighbors = num_neighbors
        # Select only the number of nns specified
        self.nn_mapping = np.load(nn_mapping)[:, :num_neighbors]
        # len(self.train_text_memmap_dataset.src_indexed_dataset) is oversampled 
        assert len(self.train_text_memmap_dataset.src_indexed_dataset) == len(self.nn_mapping)
    
    def __len__(self):
        return len(self.train_text_memmap_dataset)
    
    def __getitem__(self, idx):
        # ! Convert this to a dictionary i guess
        # Output format is a tuple of ((src, tgt), [list of neighbors in (src,tgt) format]))])
        return {
            'text_enc': self.train_text_memmap_dataset[idx]['text_enc'],
            'text_dec': self.train_text_memmap_dataset[idx]['text_dec'],
            'labels': self.train_text_memmap_dataset[idx]['labels'],
            'nn_src_list': [self.retrieval_text_memmap_dataset[self.nn_mapping[idx][i]]['text_enc'] for i in range(self.num_neighbors)],
            'nn_tgt_list': [self.retrieval_text_memmap_dataset[self.nn_mapping[idx][i]]['text_dec'] for i in range(self.num_neighbors)]
        }

    def collate_fn(self, batch):
        # This is where the batch is being created so modify this for retrieval case
        output = self.train_text_memmap_dataset.collate_fn(batch)

        retrieval_encoder_input_src_list = []
        retrieval_encoder_input_tgt_list = []
        retrieval_encoder_input_src_mask_list = []
        retrieval_encoder_input_tgt_mask_list = []

        for idx in range(self.num_neighbors):
            retrieval_input_src = [item['nn_src_list'][idx] for item in batch]
            retrieval_input_tgt = [item['nn_tgt_list'][idx] for item in batch]

            if isinstance(retrieval_input_src[0], np.ndarray):
                retrieval_input_src = [x.tolist() for x in retrieval_input_src]

            if isinstance(retrieval_input_tgt[0], np.ndarray):
                retrieval_input_tgt = [x.tolist() for x in retrieval_input_tgt]

            max_retrieval_input_tgt_length = max([len(item) for item in retrieval_input_tgt]) if retrieval_input_tgt else 0
            max_retrieval_input_src_length = max([len(item) for item in retrieval_input_src]) if retrieval_input_src else 0

            retrieval_input_src = [item + [self.retrieval_text_memmap_dataset.src_tokenizer.pad_id] * (max_retrieval_input_src_length - len(item)) for item in retrieval_input_src]
            retrieval_input_tgt = [item + [self.retrieval_text_memmap_dataset.tgt_tokenizer.pad_id] * (max_retrieval_input_tgt_length - len(item)) for item in retrieval_input_tgt]

            retrieval_input_src = torch.LongTensor(retrieval_input_src)
            retrieval_input_tgt = torch.LongTensor(retrieval_input_tgt)

            retrieval_input_src_mask = (retrieval_input_src != self.retrieval_text_memmap_dataset.src_tokenizer.pad_id).long()
            retrieval_input_tgt_mask = (retrieval_input_tgt != self.retrieval_text_memmap_dataset.tgt_tokenizer.pad_id).long()

            retrieval_encoder_input_src_list.append(retrieval_input_src)
            retrieval_encoder_input_tgt_list.append(retrieval_input_tgt)

            retrieval_encoder_input_src_mask_list.append(retrieval_input_src_mask)
            retrieval_encoder_input_tgt_mask_list.append(retrieval_input_tgt_mask)

        # TODO: This is a hack, fix this. Figure out how to do for multiple neighbors
        output['nn_src_list'] = retrieval_encoder_input_src_list[0]
        output['nn_tgt_list'] = retrieval_encoder_input_tgt_list[0]
        output['nn_src_mask_list'] = retrieval_encoder_input_src_mask_list[0]
        output['nn_tgt_mask_list'] = retrieval_encoder_input_tgt_mask_list[0]

        return output