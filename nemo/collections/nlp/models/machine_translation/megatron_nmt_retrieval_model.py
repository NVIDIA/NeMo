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
from omegaconf import DictConfig, ListConfig, OmegaConf
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
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin, NLPSaveRestoreConnector
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils, timers

from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)

try:
    from apex.transformer import parallel_state, tensor_parallel
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

        # TODO: load configs params
        
        # load the perceiver
        # TODO: See if this is correct? 
        # TODO: Do I need to call apex's build_model here?
        # TODO: Need to extract encoder from MegatronBARTModel which is perceiver encoder and transformer decoder
        self.retrieval_encoder = MegatronBARTModel.restore_from(
                    cfg.get("retrieval_encoder"),
                    trainer=trainer, 
                    save_restore_connector=NLPSaveRestoreConnector(),
                )
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
            decoder_tokenizer=self.retrieval_encoder.tokenizer
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
        self._validation_without_neighbors_ds = MTEncDecModel._setup_eval_dataset_from_config(
            cfg=self._cfg.validation_ds,
            multilingual=self.multilingual,
            multilingual_ids=self.multilingual_ids,
            encoder_tokenizer=self.encoder_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
        )
        self._validation_ds = RetrievalSeq2Seq(
            self._validation_without_neighbors_ds,
            self._retrieval_ds,
            self._cfg.validation_ds.nn_mapping,
            self._cfg.validation_ds.num_neighbors,
        )

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
            self._test_ds = RetrievalSeq2Seq(
                self._test_without_neighbors_ds,
                self._retrieval_ds,
                self._cfg.test_ds.nn_mapping,
                self._cfg.test_ds.num_neighbors,
            )
            

    def process_global_batch_for_text_translation_datasets(self, batch):
        #TODO: See how to modify this for retrievals
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


    def training_step(self, batch, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            Batch should be a list of microbatches and those microbatches should on CPU.
            Microbatches are then moved to GPU during the pipeline.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()
        # we prepare the micro batches for the apex fwd/bwd function
        batch_for_pipeline = self.process_global_batch(batch)
        encoder_seq_length = batch_for_pipeline[0].size(1) 
        decoder_seq_length = batch_for_pipeline[1].size(1)
        tensor_shape = [encoder_seq_length, get_micro_batch_size(), self.cfg.encoder.hidden_size]

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            losses_reduced_per_micro_batch = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.enc_dec_model,
                forward_only=False,
                tensor_shape=tensor_shape,
                decoder_sequence_length=decoder_seq_length,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            )
        else:
            # no pipeline parallelism so we reduce grads asynchronously
            if self.megatron_amp_o2:
                custom_sync_context_handler = self._optimizer.no_sync
            else:
                # TODO: enable async grad all reduce for O1/autocast mixed precision training
                custom_sync_context_handler = None
            losses_reduced_per_micro_batch = forward_backward_no_pipelining(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.enc_dec_model,
                forward_only=False,
                tensor_shape=tensor_shape,
                decoder_sequence_length=decoder_seq_length,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
                custom_sync_context_handler=custom_sync_context_handler,
            )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()

        if self.megatron_amp_o2:
            # when using pipeline parallelism grads must be reduced after the pipeline (not asynchronously)
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we allreduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # when using pipeline parallelism, we need keep the word and position embeddings in sync
            self.allreduce_word_and_position_embeddings()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss_mean
        
    def process_global_batch(self, global_batch):
        # ! modify for retrieval
        # ! See what the collate function of retrievalseq2seq returns
        return [
            global_batch["text_enc"],
            global_batch["text_dec"],
            global_batch["loss_mask"],
            global_batch["labels"],
            global_batch["enc_mask"],
            global_batch["dec_mask"],
            global_batch["nn_src_list"],
            global_batch["nn_tgt_list"],
        ]

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func_perceiver(batch, model):
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
        
        def fwd_output_and_loss_func_concat(batch, model):
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
        
        if self.cfg.retriever.get('type', False) == 'text':
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
        nn_mmaping,
        num_neighbors = 10,
    ):
        self.train_text_memmap_dataset = train_text_memmap_dataset
        self.retrieval_text_memmap_dataset = retrieval_text_memmap_dataset
        self.nn_mmaping = nn_mmaping

        assert len(self.train_text_memmap_dataset) == len(self.nn_mmaping)

        self.num_neighbors = num_neighbors
        # Select only the number of nns specified
        self.nn_list = np.load(nn_mmaping)[:, :num_neighbors]
    
    def __len__(self):
        return len(self.train_text_memmap_dataset)
    
    def __getitem__(self, idx):
        # ! Convert this to a dictionary i guess
        # Output format is a tuple of ((src, tgt), [list of neighbors in (src,tgt) format]))])
        return {
            'text_enc': self.train_text_memmap_dataset[idx]['text_enc'],
            'text_dec': self.train_text_memmap_dataset[idx]['text_dec'],
            'labels': self.train_text_memmap_dataset[idx]['labels'],
            'nn_src_list': [self.retrieval_text_memmap_dataset[self.nn_mmaping[idx][i]]['text_enc'] for i in range(self.num_neighbors)],
            'nn_tgt_list': [self.retrieval_text_memmap_dataset[self.nn_mmaping[idx][i]]['text_dec'] for i in range(self.num_neighbors)]
        }

    def collate_fn(self, batch):
        # ! This is where the batch is being created so modify this for retrieval case
        # ! See if you can use dataset.collate_fn? 
        # 1 what batch comes and what goes out? Where is the collate_fn being called? Any other inherited functions to modify?
        enc_query = [item['text_enc'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]

        if isinstance(enc_query[0], np.ndarray):
            enc_query = [x.tolist() for x in enc_query]

        if isinstance(dec_input[0], np.ndarray):
            dec_input = [x.tolist() for x in dec_input]

        if isinstance(labels[0], np.ndarray):
            labels = [x.tolist() for x in labels]

        max_dec_input_length = max([len(item) for item in dec_input]) if dec_input else 0
        max_enc_query_length = max([len(item) for item in enc_query]) if enc_query else 0
        max_label_length = max([len(item) for item in labels]) if labels else 0

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.src_tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        dec_input = [item + [self.tgt_tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tgt_tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]

        enc_query = torch.LongTensor(enc_query)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)

        enc_mask = (enc_query != self.src_tokenizer.pad_id).long()
        dec_mask = (dec_input != self.tgt_tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'loss_mask': loss_mask,
            #! this might become stale later when we add nearest neighbors
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            # TODO retrieval add
            'nn_src_list': [item['nn_src_list'] for item in batch],
            'nn_tgt_list': [item['nn_tgt_list'] for item in batch]
        }