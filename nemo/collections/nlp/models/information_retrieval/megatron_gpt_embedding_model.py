import itertools

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.information_retrieval.gpt_embedding_dataset import GPTEmbeddingDataset
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
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


class MegatronGPTEmbeddingModel(MegatronGPTSFTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.temperature = self.cfg.get('temperature', 1.0)
        self.num_soft_negatives = self.cfg.get('num_soft_negatives', 0)
        self.use_all_possible_negatives = self.cfg.get("use_all_possible_negatives", False)

    def model_provider_func(self, pre_process, post_process):
        # (@adithyare) We need post_process to be False to get hidden states in the loss_func
        return super().model_provider_func(pre_process, post_process=False)

    def _build_dataset(self, data_cfg, is_train=True):
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"SFT train/validation datasets must be provided as a list of individual JSONL files.")

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

        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            dataset = GPTEmbeddingDataset(
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
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets

    def training_step_fwd_bwd_step_call(self, dataloader_iter, batch_idx, forward_only):
        loss_mean, non_loss_tensors = self.fwd_bwd_step(dataloader_iter, batch_idx, forward_only)
        avg_pos_cs = torch.tensor(non_loss_tensors['avg_pos_cs']).mean().item()
        avg_neg_cs = torch.tensor(non_loss_tensors['avg_neg_cs']).mean().item()
        self.log("avg_pos_cs", avg_pos_cs, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log("avg_neg_cs", avg_neg_cs, prog_bar=True, rank_zero_only=True, batch_size=1)
        return loss_mean
    
    def inference_step_validation_call(self, batch, batch_idx, data_cfg, dataloader_idx=0):
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        loss, non_loss_tensors = self.local_validation_step(itertools.chain([batch]), batch_idx)
        outputs = {
            'loss': loss,
            'metadata': metadata,  # [dict]
            'q_hs': non_loss_tensors['query_hs'],  # [batch_size, hidden_size]
            'd_hs': non_loss_tensors['doc_hs'],  # [batch_size, hidden_size]
            'avg_pos_cs': non_loss_tensors['avg_pos_cs'],
            'avg_neg_cs': non_loss_tensors['avg_neg_cs'],
        }
        return outputs

    def gather_and_maybe_write_predictions(self, output, data_cfg, mode, dataloader_idx=0):
        gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            gathered_outputs,
            [{'q_hs': x['q_hs'], 'd_hs': x['d_hs'], 'metadata': x['metadata'], 'avg_pos_cs': x['avg_pos_cs'], 'avg_neg_cs': x['avg_neg_cs']} for x in output],
            group=parallel_state.get_data_parallel_group(),
        )

        # Remove duplicate examples due to distributed sampler.
        deduplicated_outputs = {
            'q_hs': [],
            'd_hs': [],
            'avg_pos_cs': [],
            'avg_neg_cs': [],
            'metadata': [],
        }
        total_size = 0
        for rank in range(0, parallel_state.get_data_parallel_world_size()):
            for batch in gathered_outputs[rank]:
                for q_hs, d_hs, metadata, avg_pos_cs, avg_neg_cs in zip(batch['q_hs'], batch['d_hs'], batch['metadata'], batch['avg_pos_cs'], batch['avg_neg_cs']):
                    total_size += 1
                    if not metadata.get("__AUTOGENERATED__", False):
                        deduplicated_outputs['q_hs'].append(q_hs)
                        deduplicated_outputs['d_hs'].append(d_hs)
                        deduplicated_outputs['avg_pos_cs'].append(avg_pos_cs)
                        deduplicated_outputs['avg_neg_cs'].append(avg_neg_cs)
                        deduplicated_outputs['metadata'].append(metadata)
                    else:
                        logging.info(f"skipping autogenerated example example...")

        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        assert metric_name == "loss", "Only loss is supported for now."
        avg_pos_cs = torch.tensor(deduplicated_outputs['avg_pos_cs']).mean().item()
        avg_neg_cs = torch.tensor(deduplicated_outputs['avg_neg_cs']).mean().item()
        self.log('val_avg_pos_cs', avg_pos_cs, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.log('val_avg_neg_cs', avg_neg_cs, prog_bar=True, rank_zero_only=True, batch_size=1)

        # Write predictions to file
        if self.global_rank == 0 and data_cfg.get("write_embeddings_to_file", False):
            logging.info(
                f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['metadata'])}"
            )

            # Check if the user provided a prefix path to the file(s) they want to write.
            if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                raise ValueError(
                    f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                )
            filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
            self.write_embeddings_to_file(
                deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}"
            )
        return deduplicated_outputs, total_size

    def write_embeddings_to_file(self, outputs, output_file_path):
        q_hs = torch.cat(outputs['q_hs'], dim=0)
        d_hs = torch.cat(outputs['d_hs'], dim=0)
        q_hs_npy = q_hs.detach().cpu().numpy()
        d_hs_npy = d_hs.detach().cpu().numpy()
        np.save(output_file_path + "_query.npy", q_hs_npy)
        np.save(output_file_path + "_doc.npy", d_hs_npy)
        return True

    def local_validation_step(self, dataloader_iter, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Check if iterator is exhausted
        dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        if done:
            return
        mode = 'test' if self.trainer.testing else 'val'
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.eval()

        loss, non_loss_tensors = self.fwd_bwd_step(dataloader_iter, batch_idx, True)

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.train()
        self.validation_step_outputs.append(loss) if mode == 'val' else self.test_step_outputs.append(loss)
        return loss, non_loss_tensors

    def contrast_all_possible_negatives(self, pos_doc_hs, neg_doc_hs, query_hs, bs):
        all_doc_hs = torch.cat([pos_doc_hs, neg_doc_hs], dim=0)  # (2bs) x hidden_size
        all_cs = torch.mm(query_hs, all_doc_hs.transpose(0, 1)) * (1.0 / self.temperature)  # (bs) x (2bs)

        select_cs = (
            torch.cat([torch.eye(bs), torch.zeros(bs, bs)], dim=1).type_as(all_cs).long()
        )  # FIXME: TODO: should not be recomputed every time (@adithyare)
        diag_positions = torch.where(select_cs == 1)  # FIXME: TODO: should not be recomputed every time (@adithyare)
        off_diag_positions = torch.where(
            select_cs == 0
        )  # FIXME: TODO: should not be recomputed every time (@adithyare)

        pos_cs = all_cs[diag_positions[0], diag_positions[1]]  # collects the top diagonal elements
        neg_cs = all_cs[off_diag_positions[0], off_diag_positions[1]]  # collects all the other elements
        cs = torch.cat([pos_cs.unsqueeze(1), neg_cs.repeat(bs, 1)], dim=1)
        return cs

    def contrast_with_sampled_negatives(self, pos_doc_hs, neg_doc_hs, query_hs, bs):
        assert (
            self.num_soft_negatives < query_hs.shape[0]
        ), f"Batch size {bs} is not large enough for {self.num_soft_negatives} soft negatives"

        # (@adithyare) sample soft negatives using a multinomial distribution
        soft_neg_samples = torch.ones((bs, bs)).type_as(query_hs)
        # (@adithyare) ensure the diagonal is zero, i.e. don't sample the same document as a soft negative
        soft_neg_samples.fill_diagonal_(0.0)
        soft_neg_samples = torch.multinomial(soft_neg_samples, self.num_soft_negatives, replacement=False)

        all_cs = torch.mm(query_hs, pos_doc_hs.transpose(0, 1)) * (1.0 / self.temperature)
        pos_cs_diag = torch.diag(all_cs)
        soft_neg_cs = all_cs[soft_neg_samples][:, :, 0]  # (@adithyare) can be made more efficient?
        neg_cs = torch.nn.functional.cosine_similarity(query_hs, neg_doc_hs, dim=-1) * (1.0 / self.temperature)
        cs = torch.cat([pos_cs_diag.unsqueeze(1), neg_cs.unsqueeze(1), soft_neg_cs], dim=1)
        return cs

    def constrast_with_hard_negatives(self, pos_doc_hs, neg_doc_hs, query_hs, bs):
        pos_cs = torch.mm(query_hs, pos_doc_hs.transpose(0, 1)).diag() * (1.0 / self.temperature)
        neg_cs = torch.mm(query_hs, neg_doc_hs.transpose(0, 1)).diag() * (1.0 / self.temperature)
        cs = torch.cat([pos_cs.unsqueeze(1), neg_cs.unsqueeze(1)], dim=1)
        return cs

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        idx = torch.arange(output_tensor.shape[1], device=output_tensor.device)
        eos_tensors = output_tensor[loss_mask, idx, :]
        bs = eos_tensors.shape[0] // 3
        query_hs = eos_tensors[::3, :]  # every third tensor is a query (bs x hidden_size)
        pos_doc_hs = eos_tensors[1::3, :]  # every third tensor is a positive doc (bs x hidden_size)
        neg_doc_hs = eos_tensors[2::3, :]  # every third tensor is a negative doc (bs x hidden_size)
        query_hs = query_hs / torch.norm(query_hs, dim=-1, keepdim=True)
        pos_doc_hs = pos_doc_hs / torch.norm(pos_doc_hs, dim=-1, keepdim=True)
        neg_doc_hs = neg_doc_hs / torch.norm(neg_doc_hs, dim=-1, keepdim=True)

        if self.use_all_possible_negatives:
            cs = self.contrast_all_possible_negatives(pos_doc_hs, neg_doc_hs, query_hs, bs)
        elif self.num_soft_negatives > 0:
            cs = self.contrast_with_sampled_negatives(pos_doc_hs, neg_doc_hs, query_hs, bs)
        else:
            cs = self.constrast_with_hard_negatives(pos_doc_hs, neg_doc_hs, query_hs, bs)

        avg_pos_cs = cs[:, 0].mean().item()
        avg_neg_cs = cs[:, 1:].mean().item()
        #print("pos_cs", avg_pos_cs, "neg_cs", avg_neg_cs)
        loss = torch.nn.functional.cross_entropy(cs, torch.zeros(bs).type_as(cs).long())
        cp_size = self.cfg.get('context_parallel_size', 1)
        if cp_size > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss, query_hs.clone().detach(), pos_doc_hs.clone().detach(), avg_pos_cs, avg_neg_cs
