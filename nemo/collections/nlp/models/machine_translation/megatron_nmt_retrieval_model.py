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

from typing import List
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.models.language_modeling.megatron_bart_model import MegatronBARTModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging, AppState, logging, timers

from nemo.collections.nlp.modules.common.megatron.utils import (
    build_position_ids,
    average_losses_across_data_parallel_group
)

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator
    )
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['MegatronNMTRetrievalModel']

class MegatronNMTRetrievalModel(MegatronNMTModel):
    """
    Proposed Retrieval-Model for NMT by conditioning on Perceiver embeddings
    Things to implement:
    > condition on perceiver embeddings during val/test
    > condition on raw text only
    > Figure out max_seq_length constraint is satisfied or not

    Second step
    > Implement cross attention style
    > Do monolingual stuff using retro
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        
        self.num_neighbors = self.cfg.train_ds.num_neighbors
        self.num_neighbors_eval = self.cfg.validation_ds.num_neighbors

        if self.cfg.retriever.type == 'perceiver' or self.cfg.retriever.type == 'perceiver_text':
            # Load the perceiver model
            pretrained_cfg = MegatronBARTModel.restore_from(cfg.retriever.get("encoder_path"), trainer=trainer, return_config=True)
            OmegaConf.set_struct(pretrained_cfg, True)
            with open_dict(pretrained_cfg):
                pretrained_cfg.masked_softmax_fusion = False
                pretrained_cfg.global_batch_size = cfg.global_batch_size
                pretrained_cfg.micro_batch_size = cfg.micro_batch_size
                pretrained_cfg.global_batch_per_gpu = cfg.global_batch_per_gpu
                pretrained_cfg.precision = cfg.retriever.precision
            self.retrieval_encoder = MegatronBARTModel.restore_from(
                        cfg.retriever.get("encoder_path"),
                        trainer=trainer, 
                        override_config_path=pretrained_cfg,
                        save_restore_connector=NLPSaveRestoreConnector(),
                    )
            del self.retrieval_encoder.enc_dec_model.enc_dec_model.decoder
            self.retrieval_encoder.precision = cfg.retriever.precision
            self.retrieval_encoder.freeze()
            # Using perceiver tokenizer only
            if self.cfg.retriever.type == 'perceiver':
                # Need to do this because the 
                self.encoder_tokenizer = self.retrieval_encoder.tokenizer
                self.decoder_tokenizer = self.retrieval_encoder.tokenizer

    def build_train_valid_test_datasets(self):
        """Builds the train, validation, and test datasets."""
        self._train_without_neighbors_ds = self.build_memmap_dataset_from_config(self._cfg.train_ds)
        # Retrieval dataset is not oversampled so we won't give max_num_samples
        self._retrieval_ds = self.build_memmap_dataset_from_config(
            self._cfg.retrieval_ds,
            encoder_tokenizer=self.retrieval_encoder.tokenizer if self._cfg.retriever.type =='perceiver' else None,
            decoder_tokenizer=self.retrieval_encoder.tokenizer if self._cfg.retriever.type =='perceiver' else None,
            no_oversample=True,
            )
        self._train_ds = RetrievalSeq2Seq(
            self._train_without_neighbors_ds,
            self._retrieval_ds,
            self._cfg.train_ds.nn_mapping,
            self._cfg.train_ds.num_neighbors,
            text_append_mode=self._cfg.retriever.type =='text',
            copy_prob=self._cfg.retriever.copy_prob,
            mask_prob=self._cfg.retriever.mask_prob,
        )

        if self._cfg.validation_ds.get("dataset_type", "text_memmap") != "text_memmap":
            raise ValueError(f"Validation dataset must be text_memmap for RetrievalNMT models, found {self._cfg.validation_ds.dataset_type}")
        
        #! FIX tHIS
        self._validation_without_neighbors_ds = self.build_memmap_dataset_from_config(
            self._cfg.validation_ds,
            no_oversample=True
        )
        
        if isinstance(self._validation_without_neighbors_ds, List):
            self._validation_ds = []
            for idx, ds in enumerate(self._validation_without_neighbors_ds):
                self._validation_ds.append(RetrievalSeq2Seq(
                    ds,
                    self._retrieval_ds,
                    self._cfg.validation_ds.nn_mapping[idx],
                    self._cfg.validation_ds.num_neighbors,
                    text_append_mode=self._cfg.retriever.type =='text', 
                ))
        else:
            self._validation_ds = RetrievalSeq2Seq(
                self._validation_without_neighbors_ds,
                self._retrieval_ds,
                self._cfg.validation_ds.nn_mapping,
                self._cfg.validation_ds.num_neighbors,
                text_append_mode=self._cfg.retriever.type == 'text',
            )
        # Test data config is optional.
        if hasattr(self._cfg, 'test_ds'):
            if self._cfg.test_ds.get("dataset_type", "text_memmap") != "text_memmap":
                raise ValueError(f"Test dataset type must be 'text_memmap', found {self._cfg.test_ds.dataset_type}")
            
            self._test_without_neighbors_ds = self.build_memmap_dataset_from_config(
                self._cfg.test_ds,
                no_oversample=True
            )
            
            if isinstance(self._test_without_neighbors_ds, List):
                self._test_ds = []
                for idx,ds in enumerate(self._test_without_neighbors_ds):
                    self._test_ds.append(RetrievalSeq2Seq(
                        ds,
                        self._retrieval_ds,
                        self._cfg.test_ds.nn_mapping[idx],
                        self._cfg.test_ds.num_neighbors,
                        text_append_mode=self._cfg.retriever.type == 'text',
                    ))
            else:
                self._test_ds = RetrievalSeq2Seq(
                    self._test_without_neighbors_ds,
                    self._retrieval_ds,
                    self._cfg.test_ds.nn_mapping,
                    self._cfg.test_ds.num_neighbors,
                    text_append_mode=self._cfg.retriever.type == 'text',
                )

    def process_global_batch_for_text_translation_datasets(self, batch):
        return self.process_global_batch(batch)

    def process_global_batch(self, global_batch):
        output = [
            global_batch["text_enc"],
            global_batch["text_dec"],
            global_batch["loss_mask"],
            global_batch["labels"],
            global_batch["enc_mask"],
            global_batch["dec_mask"]
        ]
        if global_batch.get("nn_src_list", None) is None:
            return output
        else:
            for idx in range(self.num_neighbors):
                output.append(global_batch["nn_src_list"][idx])
                output.append(global_batch["nn_tgt_list"][idx])
                output.append(global_batch["nn_src_mask_list"][idx])
                output.append(global_batch["nn_tgt_mask_list"][idx])
            return output

    def get_forward_output_and_loss_func(self, return_logic=False):
        """
            This function is used to create a closure that is passed to the apex fwd/bwd functions.
            Overrided from enc_dec_model.py
        """
        def retrieval_append_logic(batch, model, eval=False):
            """
            1. Get the embeddings from the embedding matrix and add positional embeddings
            2. Get the perceiver encodings from self.retriever. Embedding dropout? 
            3. Concatenate the two
            4. Pass it to the model forward using encoder inputs option
            5. Figure out how to do batching here properly
            use_src_txt: If True, use the source text as the input to the encoder. Else, use the perceiver embeddings
            """
            use_src_txt = self.cfg.retriever.get('type', 'perceiver') == 'perceiver_text'
            batch = [x.cuda(non_blocking=True) for x in batch]
            
            (
                encoder_input_ids,
                decoder_input_ids, 
                loss_mask, 
                lm_labels, 
                encoder_attn_mask, 
                decoder_attn_mask
            ) = batch[0:6]
            
            if use_src_txt:
                # Get the embeddings from the embedding matrix and add positional embeddings
                if model.encoder_cfg.get("position_embedding_type", "learned_absolute") != 'relative':
                    enc_position_ids = build_position_ids(encoder_input_ids)
                else:
                    enc_position_ids = None
                enc_input = model.encoder_embedding(encoder_input_ids, enc_position_ids, token_type_ids=None)
            
            enc_input_append = []
            
            if not eval and np.random.uniform(0,1) < self.cfg.retriever.get('mask_prob', 0.0) :
                masking = True
            else:
                masking = False
            
            if masking is False:
                # Always add NNs in eval or if sampled in (0,1) is more than threshold during training
                shuffled_idxes = list(range(self.num_neighbors))
                random.shuffle(shuffled_idxes)
                for idx in shuffled_idxes:
                    # Iterate over the neighbors and get the embeddings
                    # TODO: Can club all those calls into one but at the risk of extra padding
                    nn_src_ids = batch[6 + (idx * 4) + 0]
                    nn_tgt_ids = batch[6 + (idx * 4) + 1]
                    nn_src_mask = batch[6 + (idx * 4) + 2]
                    nn_tgt_mask = batch[6 + (idx * 4) + 3]
                    
                    # Compute perceiver embeddings. Dim is [B x S x H]
                    latent_src = self.retrieval_encoder.encode(nn_src_ids, nn_src_mask)
                    latent_tgt = self.retrieval_encoder.encode(nn_tgt_ids, nn_tgt_mask)
                    enc_input_append.append(latent_src)
                    enc_input_append.append(latent_tgt)

            if use_src_txt:
                enc_input_append.append(enc_input.transpose(0,1))
                # join latents with the encoder input embeddings
                # Concatenate along the sequence dimension
                enc_input = torch.cat(enc_input_append, dim=1)
                
                # Modify the mask to account for the new sequence length
                prepend_length = 2 * self.num_neighbors * latent_src.shape[1]
                mask_prepend = torch.ones((encoder_attn_mask.shape[0], prepend_length)).to(encoder_attn_mask.device)
                encoder_attn_mask = torch.cat((mask_prepend, encoder_attn_mask), dim=1)
            else:
                latent_src = self.retrieval_encoder.encode(encoder_input_ids, encoder_attn_mask)
                enc_input_append.append(latent_src)
                enc_input = torch.cat(enc_input_append, dim=1)
                if masking is False:
                    prepend_length = (1 + 2 * self.num_neighbors) * latent_src.shape[1] 
                else:
                    prepend_length = (1) * latent_src.shape[1]
                encoder_attn_mask = torch.ones((encoder_attn_mask.shape[0], prepend_length)).to(encoder_attn_mask.device)
            
            return (
                encoder_attn_mask,
                decoder_input_ids,
                decoder_attn_mask,
                lm_labels,
                enc_input, 
                encoder_input_ids, # might be redundant
                loss_mask
            )
        
        def fwd_output_and_loss_func_perceiver(batch, model):
            (
            encoder_attn_mask,
            decoder_input_ids,
            decoder_attn_mask,
            lm_labels,
            enc_input,
            _,
            loss_mask
            ) = retrieval_append_logic(batch, model)
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
        
        if return_logic:
            return retrieval_append_logic
        if self.cfg.retriever.get('type', False) == 'text':
            return super(MegatronNMTModel, self).get_forward_output_and_loss_func()
        elif self.cfg.retriever.get('type', False) == 'perceiver':
            return fwd_output_and_loss_func_perceiver
        else:
            raise NotImplementedError
    
    def eval_step(self, batch, batch_idx, dataloader_idx, no_ret=False):
        """"
        In parent class: batch is bucketed using OLD NMT
        Here batch comes from collate of RetrievalSeq2Seq and is a dict and is properly formatted.
        Batch size i think was manually configured somwhere to 32 in setup_eval_dataloader
        """
        # Eval step requires text datasets so we need to reconfigure MBS on each batch.
        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=batch['text_enc'].size(0) * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=batch['text_enc'].size(0),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        # This returns the averaged loss across data-parallel groups.
        # gotta call super of super
        reduced_loss = super(MegatronNMTModel, self).validation_step(batch, batch_idx)

        if self.multilingual:
            source_processor = self.source_processor_list[dataloader_idx]
            target_processor = self.target_processor_list[dataloader_idx]
        else:
            source_processor = self.source_processor
            target_processor = self.target_processor

        # Do the retrieval append here
        if self.cfg.retriever.get('type', 'perceiver') == 'text':
            tokens_enc, labels, enc_mask = batch['text_enc'], batch['labels'], batch['enc_mask']
            predicted_tokens_ids, _ = self.decode(
                tokens_enc,
                enc_mask,
                min(self._cfg.train_ds.max_seq_length, tokens_enc.size(1)
                + self._cfg.max_generation_delta),  # Generate up to src-length + max generation delta. TODO: Implement better stopping when everything hits <EOS>.
                tokenizer=self.decoder_tokenizer,
            )
            encoder_inputs = self.postprocess_outputs(
                outputs=tokens_enc, tokenizer=self.encoder_tokenizer, processor=source_processor,
            )
        elif self.cfg.retriever.get('type', 'perceiver') == 'perceiver' or self.cfg.retriever.get('type', 'perceiver') == 'perceiver_text':
            retrieval_append_logic = self.get_forward_output_and_loss_func(return_logic=True)
            batch = self.process_global_batch(batch)
            (
                encoder_attn_mask,
                _,
                _,
                labels, # same as lm_labels
                enc_input, 
                tokens_enc, # same as encoder_input_ids
                _
            ) = retrieval_append_logic(batch, self.enc_dec_model, eval=True)

            predicted_tokens_ids, _ = self.decode(
                tokens_enc=tokens_enc, # will be bypassed
                enc_mask=encoder_attn_mask,
                encoder_input=enc_input, # bypasses tokens
                num_tokens_to_generate=tokens_enc.size(1)
                + self._cfg.max_generation_delta,  # Generate up to src-length + max generation delta. TODO: Implement better stopping when everything hits <EOS>.
                tokenizer=self.decoder_tokenizer,
            )

            encoder_inputs_with_retrieval = [self.postprocess_outputs(
            outputs=tokens_enc, tokenizer=self.encoder_tokenizer, processor=source_processor,
            )]

            for idx in range(self.num_neighbors):
                nn_src_ids = batch[6 + (idx * 4) + 0]

                encoder_inputs_with_retrieval.append(self.postprocess_outputs(
                outputs=nn_src_ids, tokenizer=self.encoder_tokenizer, processor=source_processor,
                ))
                nn_tgt_ids = batch[6 + (idx * 4) + 1]
                encoder_inputs_with_retrieval.append(self.postprocess_outputs(
                outputs=nn_tgt_ids, tokenizer=self.decoder_tokenizer, processor=target_processor,
                ))
            encoder_inputs = []
            for i in range(len(encoder_inputs_with_retrieval[0])):
                encoder_inputs.append(''.join([group[i] for group in encoder_inputs_with_retrieval]))
        else:
            raise NotImplementedError

        # Post-process the translations and inputs to log.
        preds = self.postprocess_outputs(
            outputs=predicted_tokens_ids, tokenizer=self.decoder_tokenizer, processor=target_processor,
        )
        labels = self.postprocess_outputs(
            outputs=labels, tokenizer=self.decoder_tokenizer, processor=target_processor,
        )
        return {
            'inputs': encoder_inputs,
            'translations': preds,
            'ground_truths': labels,
            'loss': reduced_loss,
        }

class RetrievalSeq2Seq(Dataset):
    """
    This class defines dataset for retrieval Seq2Seq model.    
    """

    def __init__(
        self,
        seq2seq_text_memmap_dataset,
        retrieval_text_memmap_dataset,
        nn_mapping,
        num_neighbors = 10,
        text_append_mode = False,
        copy_prob = 0.0,
        mask_prob = 0.0,
        max_seq_length = 512,
    ):
        """
        seq2seq_text_memmap_dataset (TextMemmapSequenceToSequenceDataset): The seq2seq dataset train/eval
        retrieval_text_memmap_dataset (TextMemmapSequenceToSequenceDataset): The retrieval dataset
        nn_mapping (np_array): The mapping of the seq2seq dataset to the retrieval dataset
        """
        self.seq2seq_text_memmap_dataset = seq2seq_text_memmap_dataset
        self.retrieval_text_memmap_dataset = retrieval_text_memmap_dataset
        self.num_neighbors = num_neighbors
        self.text_append_mode = text_append_mode
        self.copy_prob = copy_prob
        self.mask_prob = mask_prob
        # Correct for both 'text' and 'perceiver' modes
        self.nn_start_tag = self.retrieval_text_memmap_dataset.src_tokenizer.token_to_id('▁<extra_id_0>')
        self.nn_end_tag = self.retrieval_text_memmap_dataset.src_tokenizer.token_to_id('▁<extra_id_1>')
        self.max_seq_length = max_seq_length

        # Select only the number of nns specified
        self.nn_mapping = np.load(nn_mapping)[:, :num_neighbors]

        # len(self.seq2seq_text_memmap_dataset) is oversampled so use the src_indexed_dataset
        assert len(self.seq2seq_text_memmap_dataset.src_indexed_dataset) == len(self.nn_mapping)

    def __len__(self):
        return len(self.seq2seq_text_memmap_dataset)

    def __getitem__(self, idx):
        """"
        Returns a tuple of ((src, tgt), [list of neighbors in (src,tgt) format]))])
        In "text" mode, add the neighbors to the src neighbors are empty
        Copying is handled for all modes.
        Masking is only handled for "text" mode. FOr other needs to be implemented in forward method
        """
        data_item = self.seq2seq_text_memmap_dataset[idx]
        # Need this idx_mapped to index into retrieval set so as to maintain correct mapping
        idx_mapped = data_item['idx']
        if idx_mapped >= len(self.nn_mapping):
            raise ValueError(f'idx_mapped {idx_mapped} is greater than len(self.nn_mapping) {len(self.nn_mapping)}')

        # Construct the nearest neighbors list
        nn_src_list = []
        nn_tgt_list = []
        num_iters = self.num_neighbors
        if np.random.uniform(0,1) < self.copy_prob:
            # Add copy example
            nn_src_list.append(data_item['text_enc'].tolist())
            # 'text_dec' doesn't have eos
            nn_tgt_list.append(data_item['text_dec'].tolist() + [self.retrieval_text_memmap_dataset.tgt_tokenizer.eos_id])
            # if copying original example, then only need to add one less neighbor
            num_iters = self.num_neighbors - 1
        
        for i in range(num_iters):
            element = self.retrieval_text_memmap_dataset[self.nn_mapping[idx_mapped][i]]
            nn_src_list.append(element['text_enc'].tolist())
            assert self.retrieval_text_memmap_dataset.tgt_tokenizer.eos_id == element['text_enc'].tolist()[-1]
            nn_tgt_list.append(element['text_dec'].tolist() + [self.retrieval_text_memmap_dataset.tgt_tokenizer.eos_id])
        
        # If text_append_mode add the original text to the end of the neighbors and remove neighbors
        if self.text_append_mode:
            data_item['text_enc']= data_item['text_enc'].tolist()
            data_item['text_dec'] = data_item['text_dec'].tolist()
            # Add neighbors only if not masking
            if np.random.uniform(0,1) >= self.mask_prob:
                # Need to remove <eos> from src and <bos> and <eos> from the neighbors
                assert self.seq2seq_text_memmap_dataset.tgt_tokenizer.eos_id == data_item['text_enc'][-1]
                # Remove eos_id
                data_item['text_enc'] = data_item['text_enc'][:-1]
                shuffled_idxes = list(range(self.num_neighbors))
                random.shuffle(shuffled_idxes)
                for idx in shuffled_idxes:
                    data_item['text_enc'].extend([self.nn_start_tag] + nn_src_list[idx][1:-1])
                    data_item['text_enc'].extend([self.nn_end_tag] + nn_tgt_list[idx][1:-1])
                
                # Make sure to reduce to max_seq_length
                if len(data_item['text_enc']) > self.max_seq_length - 2:
                    data_item['text_enc'] = data_item['text_enc'][: self.max_seq_length - 2]
                # Add eos_id back
                data_item['text_enc'] = data_item['text_enc'] + [self.seq2seq_text_memmap_dataset.tgt_tokenizer.eos_id]
            nn_src_list = None
            nn_tgt_list = None
        assert len(data_item['text_enc']) <= self.max_seq_length
        assert len(data_item['text_dec']) <= self.max_seq_length
        return {
            'text_enc': data_item['text_enc'],
            'text_dec': data_item['text_dec'],
            'labels': data_item['labels'],
            'nn_src_list': nn_src_list,
            'nn_tgt_list': nn_tgt_list
        }

    def collate_fn(self, batch):
        output = self.seq2seq_text_memmap_dataset.collate_fn(batch)
        if self.text_append_mode:
            return output

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

        output['nn_src_list'] = retrieval_encoder_input_src_list
        output['nn_tgt_list'] = retrieval_encoder_input_tgt_list
        output['nn_src_mask_list'] = retrieval_encoder_input_src_mask_list
        output['nn_tgt_mask_list'] = retrieval_encoder_input_tgt_mask_list

        return output