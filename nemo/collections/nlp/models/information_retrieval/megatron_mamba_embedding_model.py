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

import torch
import torch.nn.functional as F
from numpy import log2
from omegaconf import DictConfig, ListConfig


from megatron.core import parallel_state
from megatron.core.models.mamba import MambaEmbeddingModel, MambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from pytorch_lightning.trainer.trainer import Trainer
from torch.distributed import all_gather as all_gather_no_backprop
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop

from nemo.collections.nlp.data.information_retrieval.gpt_embedding_dataset import GPTEmbeddingDataset
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.models.information_retrieval.megatron_gpt_embedding_model import MegatronGPTEmbeddingModel
from nemo.utils import logging


class MegatronMambaEmbeddingModel(MegatronGPTEmbeddingModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.vocab_size = cfg.get('vocab_size', 65536)
        self.mcore_gpt = True
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
        self.temperature = self.cfg.get('temperature', 1)

        self.hard_negatives_to_train = self.cfg.data.train_ds.get("hard_negatives_to_train", 4)
        self.global_inbatch_negatives = self.cfg.get("global_inbatch_negatives", True)
        self.use_all_possible_negatives = self.cfg.get("use_all_possible_negatives", False)
        self.use_inbatch_negatives = self.cfg.get("use_inbatch_negatives", True)

        self.backprop_type = self.cfg.get("backprop_type", "local")
        assert self.backprop_type in ["local", "global"], "Backprop type must be `local` or `global`"

        self.output_head_type = self.cfg.get("output_head_type", "eos_only")
        assert self.output_head_type in [
            "eos_only",
            "avg_pool",
        ], "Output head type must be `eos_only` or `avg_pool`"
        self.bidirectional_sequences = self.cfg.get("bidirectional_sequences", False)

        # Matryoshka Representation Learning
        if self.cfg.get("do_mrl", False):
            min_mrl = self.cfg.get("min_mrl_dim", int(log2(32))) - 1
            max_mrl = int(log2(self.cfg.hidden_size // 2))
            self.mrl_dims = [2**i for i in range(max_mrl, min_mrl, -1)]
        else:
            self.mrl_dims = []

        assert (
            self.cfg.get("post_process", False) is False
        ), "post_process must be False to get hidden states in the loss_func"

    def model_provider_func(self, pre_process, post_process):
        self.hybrid_override_pattern = self.cfg.get(
            'hybrid_override_pattern', "M" * self.transformer_config.num_layers
        )
        self.transformer_config.add_bias_linear = self.cfg.get('add_bias_linear', False)
        self.transformer_config.gated_linear_unit = self.cfg.get('gated_linear_unit', False)
        self.transformer_config.layernorm_epsilon = self.cfg.get('layernorm_epsilon', 1e-5)
        self.use_latent_attention = self.cfg.get("use_latent_attention", False)

        if self.use_latent_attention:
            model = MambaEmbeddingModel(
                config=self.transformer_config,
                max_sequence_length=self.cfg.get('encoder_seq_length', 4096),
                vocab_size=self.cfg.get('vocab_size', 65536),
                mamba_ssm_ngroups=self.cfg.get('mamba_ssm_ngroups', 8),
                mamba_stack_spec=mamba_stack_spec,
                hybrid_override_pattern=self.hybrid_override_pattern,
                pre_process=pre_process,
                post_process=False,
                num_latent_attention_vectors=self.cfg.get('num_latent_attention_vectors', 64),
                num_latent_attention_heads=self.cfg.get('num_latent_attention_heads', 4),
                latent_dim=self.cfg.get('latent_dim', 1024),
                latent_inner_dim=self.cfg.get('latent_inner_dim', 256),
                dropout=0.1,
            )
        else:
            model = MambaModel(
                config=self.transformer_config,
                max_sequence_length=self.cfg.get('encoder_seq_length', 4096),
                vocab_size=self.cfg.get('vocab_size', 65536),
                mamba_ssm_ngroups=self.cfg.get('mamba_ssm_ngroups', 8),
                mamba_stack_spec=mamba_stack_spec,
                hybrid_override_pattern=self.hybrid_override_pattern,
                pre_process=pre_process,
                post_process=False,
            )

        return model

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
            num_query_files = len(data_cfg.query_file_names) if data_cfg.query_file_names is not None else 0
            num_doc_files = len(data_cfg.doc_file_names) if data_cfg.doc_file_names is not None else 0
            num_query_samples_per_dataset = [[None]] * num_query_files
            num_doc_samples_per_dataset = [[None]] * num_doc_files

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
        if is_train:
            datasets = []
            for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
                dataset = GPTEmbeddingDataset(
                    file_path=file_path,
                    tokenizer=self.tokenizer,
                    num_hard_negatives=self.hard_negatives_to_train,
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
                    bidirectional_sequences=self.bidirectional_sequences,
                )
                datasets.append(dataset)
            if packed_sequence:
                raise NotImplementedError("Packed sequence is not supported for MegatronMambaEmbeddingModel")

            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            if data_cfg.query_file_names is None or data_cfg.doc_file_names is None:
                return []

            # breakpoint()

            query_dataset = GPTEmbeddingDataset(
                file_path=data_cfg.query_file_names[0],
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                max_num_samples=None,
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
                data_type="query",
                bidirectional_sequences=self.bidirectional_sequences,
            )
            doc_dataset = GPTEmbeddingDataset(
                file_path=data_cfg.doc_file_names[0],
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                max_num_samples=None,
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
                data_type="doc",
                bidirectional_sequences=self.bidirectional_sequences,
            )
            return [query_dataset, doc_dataset]

    def constrastive_scores(
            self, 
            pos_doc_hs, 
            neg_doc_hs, 
            query_hs, 
            bs, 
            temperature,
            num_hard_negatives,
            use_all_possible_negatives=False, 
            use_inbatch_negatives=True
            ):
        all_doc_hs = torch.cat([pos_doc_hs, neg_doc_hs], dim=0)  # ((hn+1)bs) x hidden_size
        cs = torch.mm(query_hs, all_doc_hs.transpose(0, 1))  # (bs) x ((hn+1)bs)
        pos_cs = cs[:, :bs]
        neg_cs = cs[:, bs:]
        if use_all_possible_negatives:
            labels = torch.arange(bs, device=cs.device).long()
        else:
            neg_cs = neg_cs[
                torch.arange(bs).unsqueeze(1).repeat(1,num_hard_negatives), 
                torch.arange(bs*num_hard_negatives).reshape(bs, num_hard_negatives)
                ]
            if use_inbatch_negatives:
                labels = torch.arange(bs, device=cs.device).long()      
                cs = torch.cat([pos_cs, neg_cs], dim=1)
            else:
                labels = torch.zeros(bs, device=cs.device).long()
                cs = torch.cat([pos_cs.diag().unsqueeze(1), neg_cs], dim=1)
        
        pos_cs = pos_cs.diag().clone().detach().mean()
        neg_cs = neg_cs.clone().detach().mean()

        # pos_cs = torch.mm(query_hs, pos_doc_hs.transpose(0, 1))
        # neg_cs = (
        #     torch.multiply(
        #         query_hs.unsqueeze(0).repeat(len(neg_doc_hs), 1, 1),
        #         torch.stack(neg_doc_hs),
        #     )
        #     .sum(axis=-1)
        #     .T
        # )
        # cs = torch.cat([pos_cs, neg_cs], axis=1)

        # labels = torch.tensor(range(len(cs)), dtype=torch.long, device=cs.device)
        # pos_cs = pos_cs.diag().clone().detach().mean()
        # neg_cs = neg_cs.clone().detach().mean()

        cs = cs.clamp(-1.0, 1.0)
        cs = cs / temperature
        return cs, pos_cs, neg_cs, labels

    def _gather_global_inbatch_representations(self, local_output_tensor):
        local_output_tensor = local_output_tensor.contiguous()
        if self.backprop_type == 'local':
            global_output_tensors = [
                torch.zeros_like(local_output_tensor) for _ in range(parallel_state.get_data_parallel_world_size())
            ]
            all_gather_no_backprop(
                global_output_tensors, local_output_tensor, group=parallel_state.get_data_parallel_group()
            )
            global_output_tensors[parallel_state.get_data_parallel_rank()] = local_output_tensor
            global_output_tensors = torch.cat(global_output_tensors, dim=0)

        else:
            global_output_tensors = all_gather_with_backprop(local_output_tensor)
            global_output_tensors = torch.cat(global_output_tensors, dim=0)

        return global_output_tensors

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        # if self.cfg.get('sequence_parallel', False):
            # output_tensor = gather_from_sequence_parallel_region(output_tensor)

        output_tensor = output_tensor.permute(1, 0, 2)
        if self.output_head_type == "eos_only":
            if self.bidirectional_sequences:
                fwd_output_tensor, rev_output_tensor = output_tensor.chunk(2)
                idx = torch.arange(fwd_output_tensor.shape[0], device=output_tensor.device)
                fwd_output_tensor = fwd_output_tensor[idx, loss_mask, :]
                rev_output_tensor = rev_output_tensor[idx, loss_mask, :]
                output_tensor = (fwd_output_tensor + rev_output_tensor) / 2.0

            else:
                idx = torch.arange(output_tensor.shape[0], device=output_tensor.device)
                output_tensor = output_tensor[idx, loss_mask, :]
        else:
            if self.bidirectional_sequences:
                raise NotImplementedError("Bidirectional sequences are not supported for avg_pool output head")

            lengths = loss_mask + 1
            attention_mask = torch.arange(output_tensor.shape[1], device=output_tensor.device).expand(
                len(lengths), output_tensor.shape[1]
            ) < lengths.unsqueeze(1)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_tensor.size()).float()
            sum_embeddings = torch.sum(output_tensor * input_mask_expanded, 1)

            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_tensor = sum_embeddings / sum_mask

        if self.global_inbatch_negatives and self.trainer.training:
            output_tensor = self._gather_global_inbatch_representations(output_tensor)

        if not self.trainer.training:
            return self.inference_loss_func(loss_mask, num_valid_tokens_in_ub, output_tensor)

        num_tensors_per_example = 2 + self.hard_negatives_to_train
        bs = output_tensor.shape[0] // num_tensors_per_example
        chunks = output_tensor.chunk(bs)
        query_hs = torch.stack([item[0] for item in chunks])
        pos_doc_hs = torch.stack([item[1] for item in chunks])
        # neg_doc_hs = [torch.stack([item[i + 2] for item in chunks]) for i in range(self.hard_negatives_to_train)]
        neg_doc_hs = torch.stack([item[i + 2]  for item in chunks for i in range(self.hard_negatives_to_train)])

        query_hs = F.normalize(query_hs, dim=1)
        pos_doc_hs = F.normalize(pos_doc_hs, dim=1)
        # neg_doc_hs = [F.normalize(nd, dim=1) for nd in neg_doc_hs]
        neg_doc_hs = F.normalize(neg_doc_hs, dim=1)

        cs, pos_cs, neg_cs, labels = self.constrastive_scores(
            pos_doc_hs,
            neg_doc_hs,
            query_hs,
            bs,
            self.temperature,
            self.hard_negatives_to_train,
            self.use_all_possible_negatives,
            self.use_inbatch_negatives,
        )

        loss = self.cross_entropy_loss(cs, labels)

        if self.mrl_dims:
            for dim in self.mrl_dims:
                cs_dim, _, _, _ = self.constrastive_scores(
                    pos_doc_hs[:, :dim],
                    [nd[:, :dim] for nd in neg_doc_hs],
                    query_hs[:, :dim],
                    bs,
                    self.temperature,
                )
                loss += self.cross_entropy_loss(cs_dim, labels)

        cp_size = self.cfg.get('context_parallel_size', 1)
        if cp_size > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        query_hs = query_hs.clone().detach()
        pos_doc_hs = pos_doc_hs.clone().detach()
        diff_cs = pos_cs - neg_cs
        return {
            "loss": loss,
            "query_hs": query_hs,
            "pos_doc_hs": pos_doc_hs,
            "pos_cs": pos_cs,
            "neg_cs": neg_cs,
            "diff_cs": diff_cs,
        }
