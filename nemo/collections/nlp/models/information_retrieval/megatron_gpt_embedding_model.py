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


class MegatronGPTEmbeddingModel(MegatronGPTSFTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.temperature = self.cfg.get('temperature', 1.0)
        self.num_soft_negatives = self.cfg.get('num_soft_negatives', 0)

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

    def loss_func(self, loss_mask, num_valid_tokens_in_ub, output_tensor):
        idx = torch.arange(output_tensor.shape[1], device=output_tensor.device)
        eos_tensors = output_tensor[loss_mask, idx, :]
        query_hs = eos_tensors[::3, :]
        pos_doc_hs = eos_tensors[1::3, :]
        neg_doc_hs = eos_tensors[2::3, :]
        neg_cs = torch.nn.functional.cosine_similarity(query_hs, neg_doc_hs, dim=-1) * (1.0 / self.temperature)

        if self.num_soft_negatives > 0:
            assert (
                self.num_soft_negatives < query_hs.shape[0]
            ), f"Batch size {query_hs.shape[0]} is not large enough for {self.num_soft_negatives} soft negatives"
            query_hs = query_hs / torch.norm(query_hs, dim=-1, keepdim=True)
            pos_doc_hs = pos_doc_hs / torch.norm(pos_doc_hs, dim=-1, keepdim=True)

            # (@adithyare) sample soft negatives using a multinomial distribution
            soft_neg_samples = torch.ones((query_hs.shape[0], query_hs.shape[0])).type_as(query_hs)
            # (@adithyare) ensure the diagonal is zero, i.e. don't sample the same document as a soft negative
            soft_neg_samples.fill_diagonal_(0.0)
            soft_neg_samples = torch.multinomial(soft_neg_samples, self.num_soft_negatives, replacement=False)

            all_cs = torch.mm(query_hs, pos_doc_hs.transpose(0, 1)) * (1.0 / self.temperature)
            pos_cs_diag = torch.diag(all_cs)
            soft_neg_cs = all_cs[soft_neg_samples][:, :, 0]  # (@adithyare) can be made more efficient?
            cs = torch.cat([pos_cs_diag.unsqueeze(1), neg_cs.unsqueeze(1), soft_neg_cs], dim=1)
        else:
            pos_cs = torch.nn.functional.cosine_similarity(query_hs, pos_doc_hs, dim=-1) * (1.0 / self.temperature)
            cs = torch.cat([pos_cs.unsqueeze(1), neg_cs.unsqueeze(1)], dim=1)

        loss = torch.nn.functional.cross_entropy(cs, torch.zeros(cs.shape[0], dtype=torch.long, device=cs.device))
        cp_size = self.cfg.get('context_parallel_size', 1)
        if cp_size > 1:
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())
        return loss
