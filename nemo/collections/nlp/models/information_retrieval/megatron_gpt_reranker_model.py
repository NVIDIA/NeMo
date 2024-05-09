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

import itertools
import os

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.information_retrieval.gpt_embedding_dataset import GPTRerankerDataset
from nemo.collections.nlp.models.information_retrieval.megatron_gpt_embedding_model import MegatronGPTEmbeddingModel
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.utils import logging

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False
try:

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


def listify(tensor):
    l_tensor = []
    for t in tensor:
        for rid in range(t.shape[0]):
            r = t[rid, :].unsqueeze(0).cpu()
            l_tensor.append(r)
    return l_tensor


class MegatronGPTRerankerModel(MegatronGPTEmbeddingModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.temperature = self.cfg.get('temperature', 0.02)
        self.use_all_possible_negatives = self.cfg.get("use_all_possible_negatives", True)
        self.global_inbatch_negatives = self.cfg.get("global_inbatch_negatives", True)
        assert (
            self.cfg.get("post_process", False) is False
        ), "post_process must be False to get hidden states in the loss_func"

    def model_provider_func(self, pre_process, post_process):
        # (@adithyare) We need post_process to be False to get hidden states in the loss_func
        return super().model_provider_func(pre_process, post_process=False)

    def maybe_setup_test(self):
        if (
            hasattr(self.cfg.data, 'test_ds')
            and self.cfg.data.test_ds.get('doc_file_names', None) is not None
            and self.cfg.data.test_ds.get('query_file_names', None) is not None
        ):
            self._test_dl = self.setup_eval_dataloader(self._test_ds, self.cfg.data.test_ds)
        return

    def maybe_build_test(self):
        if (
            hasattr(self.cfg.data, 'test_ds')
            and self.cfg.data.test_ds.get('file_names', None) is not None
        ):
            logging.info('Building GPT Reranker test datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._test_ds = self._build_dataset(self.cfg.data.test_ds, is_train=False)

    def _build_dataset(self, data_cfg, is_train=True):
        packed_sequence = data_cfg.get("packed_sequence", False)

        # Determine if we are using a single dataset or a list of datasets.
        if is_train:
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
            # that is of the format [weight1,file_name1,weight2,file_name2,...]
            if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                        f"Found: {data_cfg.concat_sampling_probabilities}"
                    )
                )

            if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be of the same size as file_names.",
                        f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
                    )
                )

            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * data_cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        # Check dataset max_seq_legnth and max_position_embeddings size
        if (
            self.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
            and data_cfg.max_seq_length > self.cfg.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.max_position_embeddings

        # TE requires that the first input dim is divisible by 8 and the second by 16 for fp8
        # When using sequence parallel, sequence will further be split by TP size
        pad_seq_length_to_mult = (
            8 * self.cfg.get('tensor_model_parallel_size', 1) if self.cfg.get('sequence_parallel', False) else 16
        )
        pad_seq_length_to_mult *= self.cfg.get('context_parallel_size', 1)

        datasets = []
        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            dataset = GPTRerankerDataset(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                virtual_tokens=self.virtual_tokens,
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
                truncation_method=data_cfg.get(
                    'truncation_method', 'right'
                ),  # used to choose truncation method. Options: ['random', 'left', 'right']
                special_tokens=self.cfg.data.get(
                    'chat_prompt_tokens', None
                ),  # special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
            )
            datasets.append(dataset)
        if is_train:
            if packed_sequence:
                num_train_samples_after_blend = sum(len(dataset) for dataset in datasets)
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets

    def constrastive_scores(self, pos_doc_hs, neg_doc_hs, query_hs, bs, use_all_possible_negatives=False):
        all_doc_hs = torch.cat([pos_doc_hs, neg_doc_hs], dim=0)  # (2bs) x hidden_size
        cs = torch.mm(query_hs, all_doc_hs.transpose(0, 1))  # (bs) x (2bs)
        pos_cs = cs[:, :bs].diag()
        neg_cs = cs[:, bs:].diag()
        if use_all_possible_negatives:
            labels = torch.arange(bs, device=cs.device).long()
        else:
            labels = torch.zeros(bs, device=cs.device).long()
            cs = torch.cat([pos_cs.unsqueeze(1), neg_cs.unsqueeze(1)], dim=1)
        pos_cs = pos_cs.clone().detach().mean()
        neg_cs = neg_cs.clone().detach().mean()
        return cs, pos_cs, neg_cs, labels

    def inference_loss_func(self, loss_mask, num_valid_tokens_in_ub, eos_tensors):
        hs = eos_tensors
        hs = torch.nn.functional.normalize(hs, dim=1)
        _blank = torch.zeros(1, device=hs.device, dtype=hs.dtype)[0]
        return _blank, hs, hs, _blank, _blank, _blank

    def _gather_global_inbatch_representations(self, local_eos_tensor):
        local_eos_tensor = local_eos_tensor.contiguous()
        global_eos_tensors = [
            torch.zeros_like(local_eos_tensor) for _ in range(parallel_state.get_data_parallel_world_size())
        ]
        torch.distributed.all_gather(
            global_eos_tensors, local_eos_tensor, group=parallel_state.get_data_parallel_group()
        )
        global_eos_tensors[parallel_state.get_data_parallel_rank()] = local_eos_tensor
        global_eos_tensors = torch.cat(global_eos_tensors, dim=0)
        return global_eos_tensors

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        idx = torch.arange(output_tensor.shape[1], device=output_tensor.device)
        eos_tensors = output_tensor[loss_mask, idx, :]
        if self.global_inbatch_negatives and self.trainer.training:
            eos_tensors = self._gather_global_inbatch_representations(eos_tensors)
        if not self.trainer.training:
            return self.inference_loss_func(loss_mask, num_valid_tokens_in_ub, eos_tensors)
        bs = eos_tensors.shape[0] // 3
        query_hs = eos_tensors[::3, :]  # every third tensor is a query (bs x hidden_size)
        pos_doc_hs = eos_tensors[1::3, :]  # every third tensor is a positive doc (bs x hidden_size)
        neg_doc_hs = eos_tensors[2::3, :]  # every third tensor is a negative doc (bs x hidden_size)

        query_hs = torch.nn.functional.normalize(query_hs, dim=1)
        pos_doc_hs = torch.nn.functional.normalize(pos_doc_hs, dim=1)
        neg_doc_hs = torch.nn.functional.normalize(neg_doc_hs, dim=1)

        cs, pos_cs, neg_cs, labels = self.constrastive_scores(
            pos_doc_hs, neg_doc_hs, query_hs, bs, self.use_all_possible_negatives
        )
        cs = cs.clamp(-1.0, 1.0)
        cs = cs / self.temperature
        loss = torch.nn.functional.cross_entropy(cs, labels)

        cp_size = self.cfg.get('context_parallel_size', 1)
        if cp_size > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        query_hs = query_hs.clone().detach()
        pos_doc_hs = pos_doc_hs.clone().detach()
        diff_cs = pos_cs - neg_cs
        return loss, query_hs, pos_doc_hs, pos_cs, neg_cs, diff_cs
