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
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor
from torch import nn
from typing import Dict
import torch.nn.functional as F
import torch
from torch import Tensor
from torch import nn
from typing import Dict
import random
import os
import json
from torch.utils.data import DataLoader, Dataset
from nemo.collections.nlp.data.language_modeling.megatron import dataset_utils
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from typing import Union, Tuple, List, Dict
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import BertLMHead, post_language_model_processing, bert_extended_attention_mask
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import AppState, logging
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import BertModel
from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import get_layer_norm
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    build_position_ids,
    erf_gelu,
    get_linear_layer,
    init_method_normal,
    openai_gelu,
    parallel_lm_logits,
    scaled_init_method_normal,
)

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.tensor_parallel.layers import set_tensor_model_parallel_attributes
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()

try:
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    import logging

    from lddl.torch_mp import get_bert_pretrain_data_loader

    HAVE_LDDL = True
except (ImportError, ModuleNotFoundError):
    HAVE_LDDL = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False


class MultiplePositivesNegativesDataset(Dataset):
    """SentenceTransformer tokenizer and MultipleNegativesRankingLoss expects
        a single positive and a single hard-negative (optional) per example.
        This Dataset manages the case where there is more than one positive or negative
        available, in form of a list.
        It uses the list of positives/negatives as a queue, where for each epoch the 
        first positive/negative of the queue is used for training, after which the
        item is moved to the end of the queue.
        If num_hard_negs > 1, multiple negatives will be sampled for each example.

        Args:
            data (List[Dict[str, str]]): A list of Dict whose 
            keys are "question", "pos_doc", "neg_doc"
            num_hard_negs (int): Number of hard-negatives for each query to sample
            shuffled_negs (bool, optional): Whether the negatives per example
            needs to be shuffled in the initialization. Defaults to False.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        shuffled_negs: bool = False,
        num_hard_negs: int = 1,
        query_prefix: str = "",
        passage_prefix: str = "",
        
    ):
        self.data = data
        self.num_hard_negs = num_hard_negs
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        if shuffled_negs:
            for example in self.data:
                random.shuffle(example["neg_doc"])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        example = self.data[item]
        question = f'{self.query_prefix} {example["question"]}'.strip()
        texts = [question]

        positive = example["pos_doc"]
        if isinstance(positive, list):
            # Dequeues one positive and adds it at end of the queue
            positive = example["pos_doc"].pop(0)
            example["pos_doc"].append(positive)

        positive = f"{self.passage_prefix} {positive}".strip()
        texts.append(positive)

        negative = []
        if "neg_doc" in example:
            negative = example["neg_doc"]
            selected_negs = []
            if isinstance(negative, list):
                for _ in range(self.num_hard_negs):
                    if len(example["neg_doc"]) > 0:
                        # Dequeues a negative and adds it at end of the queue
                        negative = example["neg_doc"].pop(0)
                        selected_negs.append(negative)
                        example["neg_doc"].append(negative)
                    else:
                        # Providing empty hard-negative, for this example,
                        # so that it matches the number of hard negatives
                        # of the other examples
                        selected_negs.append("")

            else:
                selected_negs = [negative]
            selected_negs = [
                f"{self.passage_prefix} {neg}".strip() for neg in selected_negs
            ]
            texts.extend(selected_negs)
        return texts
    

##########################
# Below class is copied from SentenceTransformer library: https://github.com/UKPLab/sentence-transformers/blob/08a57b4a19ddaf7cccda51cd0c2c8af7bbc339a3/sentence_transformers/models/Normalize.py
##########################

class Normalize(nn.Module):
    """
    This layer normalizes embeddings to unit length
    """

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, features: Dict[str, Tensor]):
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features

##########################
# Below class is copied from SentenceTransformer library: https://github.com/UKPLab/sentence-transformers/blob/08a57b4a19ddaf7cccda51cd0c2c8af7bbc339a3/sentence_transformers/models/Pooling.py
##########################
    
class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but divide by sqrt(input_length).
    :param pooling_mode_weightedmean_tokens: Perform (position) weighted mean pooling, see https://arxiv.org/abs/2202.08904
    :param pooling_mode_lasttoken: Perform last token pooling, see https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
    ):
        super(Pooling, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
        ]

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ["mean", "max", "cls", "weightedmean", "lasttoken"]
            pooling_mode_cls_token = pooling_mode == "cls"
            pooling_mode_max_tokens = pooling_mode == "max"
            pooling_mode_mean_tokens = pooling_mode == "mean"
            pooling_mode_weightedmean_tokens = pooling_mode == "weightedmean"
            pooling_mode_lasttoken = pooling_mode == "lasttoken"

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken

        pooling_mode_multiplier = sum(
            [
                pooling_mode_cls_token,
                pooling_mode_max_tokens,
                pooling_mode_mean_tokens,
                pooling_mode_mean_sqrt_len_tokens,
                pooling_mode_weightedmean_tokens,
                pooling_mode_lasttoken,
            ]
        )
        self.pooling_output_dimension = pooling_mode_multiplier * word_embedding_dimension

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append("cls")
        if self.pooling_mode_mean_tokens:
            modes.append("mean")
        if self.pooling_mode_max_tokens:
            modes.append("max")
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append("mean_sqrt_len_tokens")
        if self.pooling_mode_weightedmean_tokens:
            modes.append("weightedmean")
        if self.pooling_mode_lasttoken:
            modes.append("lasttoken")

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
                .to(token_embeddings.device)
            )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
            # Any sequence where min == 1, we use the entire sequence length since argmin = 0
            values, indices = torch.min(attention_mask, 1, keepdim=False)
            gather_indices = torch.where(values == 0, indices, seq_len) - 1  # Shape [bs]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sentence_embedding": output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    
class SBertModel(BertModel):
    """
    Bert Language model.
    Model returns [seq, batch, hidden] shape
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_lm_cross_entropy=False,
        hidden_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        layernorm_epsilon=1e-5,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        masked_softmax_fusion=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        add_binary_head=True,
        skip_head=False,
        megatron_legacy=False,
        sequence_parallel=False,
        position_embedding_type='learned_absolute',
    ):
        super().__init__(config,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling,
        kv_channels,
        num_tokentypes,
        parallel_output,
        pre_process,
        post_process,
        init_method_std,
        fp16_lm_cross_entropy,
        hidden_dropout,
        precision,
        fp32_residual_connection,
        activations_checkpoint_granularity,
        activations_checkpoint_method,
        activations_checkpoint_num_layers,
        activations_checkpoint_layers_per_pipeline,
        layernorm_epsilon,
        normalization,
        transformer_block_type,
        masked_softmax_fusion,
        bias_gelu_fusion,
        bias_dropout_add_fusion,
        openai_gelu,
        onnx_safe,
        add_binary_head,
        skip_head,
        megatron_legacy,
        sequence_parallel,
        position_embedding_type,)
                
        self.pooling_add_on = Pooling(word_embedding_dimension = 1024, 
                                      pooling_mode_cls_token = False, 
                                      pooling_mode_mean_tokens = True, 
                                      pooling_mode_max_tokens = False, 
                                      pooling_mode_mean_sqrt_len_tokens = False)
        
        self.normalize_add_on = Normalize()
 

    def forward(
        self,
        bert_model_input,
        attention_mask,
        token_type_ids=None,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
    ):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            input_ids = bert_model_input
            position_ids = build_position_ids(input_ids)
        else:
            position_ids = None
            input_ids = None

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            token_type_ids=token_type_ids,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )
        

        if self.post_process and self.add_binary_head:

            lm_output, pooled_output = lm_output
        else:
            pooled_output = None

        add_on_inputs = {"token_embeddings":lm_output[0].permute(1, 0, 2), "attention_mask": attention_mask}
        lm_output = self.pooling_add_on(add_on_inputs)
        lm_output = self.normalize_add_on(lm_output)

        return lm_output['sentence_embedding']
    
 
class MegatronSBertModel(MegatronBertModel):
    """
    Megatron Bert pretraining.
    Model returns [batch, seq, hidden] shape
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):

        super().__init__(cfg, trainer=trainer)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
        softmax_temp = cfg.get('softmax_temp', 0.05)
        self.scale = 1.0/softmax_temp

    def model_provider_func(self, pre_process, post_process):
        cfg = self.cfg
        num_tokentypes = 2 if cfg.bert_binary_head else 0

        if self.mcore_bert:
            raise ValueError("mcore not supported for SBERT")
            
        else:
            model = SBertModel(
                config=self.model_parallel_config,
                vocab_size=self.padded_vocab_size,
                hidden_size=cfg.hidden_size,
                max_position_embeddings=cfg.max_position_embeddings,
                num_layers=cfg.num_layers,
                num_attention_heads=cfg.num_attention_heads,
                apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
                kv_channels=cfg.get('kv_channels', None),
                ffn_hidden_size=cfg.ffn_hidden_size,
                num_tokentypes=num_tokentypes,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=cfg.get('init_method_std', 0.02),
                fp16_lm_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
                hidden_dropout=cfg.get('hidden_dropout', 0.1),
                precision=cfg.get('precision', 16),
                fp32_residual_connection=cfg.get('fp32_residual_connection', False),
                activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
                activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
                activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
                activations_checkpoint_layers_per_pipeline=self.cfg.get(
                    'activations_checkpoint_layers_per_pipeline', None
                ),
                layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
                masked_softmax_fusion=cfg.get('masked_softmax_fusion', True),
                normalization=cfg.get('normalization', 'layernorm'),
                transformer_block_type=cfg.get('transformer_block_type', 'pre_ln'),
                bias_gelu_fusion=cfg.get('bias_gelu_fusion', True),
                bias_dropout_add_fusion=cfg.get("bias_dropout_add_fusion", True),
                onnx_safe=cfg.get('onnx_safe', False),
                add_binary_head=cfg.bert_binary_head,
                skip_head=cfg.get('skip_head', False),
                megatron_legacy=cfg.get('megatron_legacy', False),
                position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
            )

        return model


    def build_train_valid_test_datasets(self):

        train_file_path = self.cfg.data.data_prefix

        with open(train_file_path) as f:
            train_data = json.load(f)

        query_prefix = "query:"
        passage_prefix = "passage:"
        evaluation_sample_size = self.cfg.data.get("evaluation_sample_size", 0)
        hard_negatives_to_train = self.cfg.data.get("hard_negatives_to_train", 4)
        evaluation_steps = self.cfg.data.get("evaluation_steps", 0)


        #TODO @ataghibakhsh: Handle valid and test datasets better

        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None

        if train_file_path: # we don't support calculating validation loss for multiple train files 
            valid_data = None
            if evaluation_sample_size:
                if evaluation_steps == 0:
                    raise ValueError(
                        "The --evaluation_steps should be greater than 0 "
                        "when --evaluation_sample_size is set"
                    )

                if evaluation_sample_size >= len(train_data):
                    raise ValueError(
                        "The --evaluation_sample_size cannot be greater " "than train set size."
                    )

                valid_data = train_data[-evaluation_sample_size:]
                train_data = train_data[:-evaluation_sample_size]
            
            if evaluation_sample_size:
                self._validation_ds = MultiplePositivesNegativesDataset(
                    valid_data,
                    num_hard_negs=hard_negatives_to_train,
                    query_prefix=query_prefix,
                    passage_prefix=passage_prefix
                )

        self._train_ds = MultiplePositivesNegativesDataset(
            train_data,
            num_hard_negs=hard_negatives_to_train,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix
        )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building Bert datasets.')
        
        return self._train_ds, self._validation_ds, self._test_ds

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert(
            self.model
        )

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if stage == 'predict':
            return
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            if self.cfg.data.dataloader_type == "LDDL":
                self.build_LDDL_data(self.cfg.data)
                torch.distributed.barrier()
            else:
                self.build_train_valid_test_datasets()
                self.setup_training_data(self.cfg.data)
                # self.setup_validation_data(self.cfg.data)
                # self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    sync_embeddings = (
                        module.initialize_last_stage_with_word_embeddings
                        if self.mcore_bert
                        else module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                sync_embeddings = (
                    self.model.initialize_last_stage_with_word_embeddings
                    if self.mcore_bert
                    else self.model.sync_initial_word_embeddings
                )
                sync_embeddings()

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_bert', False):
            self.setup_transformer_engine_tp_groups()

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None
        
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

        dataloader.collate_fn = self.batching_collate
        
        return dataloader
    
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):

        max_seq_length=self.cfg.encoder_seq_length
        do_lower_case=self.cfg.tokenizer.get("do_lower_case", False)
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer.tokenizer(
                *to_tokenize,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=max_seq_length,
            )
        )
        return output


    def batching_collate(self, batch) :
            """
            Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
            Here, batch is a list of InputExample instances: [InputExample(...), ...]

            :param batch:
                a batch from a SmartBatchingDataset
            :return:
                a batch of tensors for the model
            """
            texts = [example for example in batch]
            sentence_features = [self.tokenize(sentence) for sentence in zip(*texts)]
            return sentence_features

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                batches = next(dataloader_iter) #for each element in batches, there should be 1 anchor, 1 positive, and n negatives                
                # In Bert dataset (like Pile), every batch has tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
                # For Sbert, we want the batch to be a list of [anchors, positives, negatives1, negatives2, ..., ] so that every of the anchors/positives/negatives are the same as the batch in pile dataset
                # batches = [anchors, positives, negatives1, negatives2]
                tokens_batch, types_batch, sentence_order_batch, loss_mask_batch, lm_labels_batch, padding_mask_batch = [], [], [], [], [], []
                for batch in batches:
                    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = (
                        batch['input_ids'].cuda(non_blocking=True),
                        batch['token_type_ids'].cuda(non_blocking=True),
                        None, # batch['is_random'].cuda(non_blocking=True),
                        None, # batch['loss_mask'].cuda(non_blocking=True),
                        None, # batch['labels'].cuda(non_blocking=True),
                        batch['attention_mask'].cuda(non_blocking=True),
                    )
                    tokens_batch.append(tokens)
                    types_batch.append(types)
                    sentence_order_batch.append(sentence_order)
                    loss_mask_batch.append(loss_mask)
                    lm_labels_batch.append(lm_labels)
                    padding_mask_batch.append(padding_mask)
            else:
                batch = next(dataloader_iter)
                if parallel_state.is_pipeline_first_stage():
                    tokens = batch['text'].cuda(non_blocking=True)
                    types = batch['types'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    loss_mask, lm_labels = None, None
                elif parallel_state.is_pipeline_last_stage():
                    loss_mask = batch['loss_mask'].cuda(non_blocking=True)
                    lm_labels = batch['labels'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    tokens, types = None, None
                else:
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    tokens, types, loss_mask, lm_labels = None, None, None, None

            if not self.cfg.bert_binary_head:
                types = None

            forward_args = [{
                "input_ids": tokens,
                "attention_mask": padding_mask,
                "lm_labels": lm_labels,
                "token_type_ids": types
            } for tokens, padding_mask, lm_labels, types in zip(tokens_batch, padding_mask_batch, lm_labels_batch, types_batch)]

            ''' if not self.mcore_bert:
                forward_args["checkpoint_activations_all_layers"] = checkpoint_activations_all_layers
                forward_args["model"] = model
                forward_args["token_type_ids"] = types
            else:
                forward_args["tokentype_ids"] = types'''
            
            output_tensor = None
            if self.mcore_bert:
                output_tensor = model(**forward_args)
            else:
                output_tensor = [self.forward(**forward_arg).permute(1,0) for forward_arg in forward_args]
   
            def loss_func(output_tensor):

                loss_dict = self.loss_func(output_tensor)
                if 'sop loss' in loss_dict:
                    lm_loss = loss_dict['lm loss']
                    sop_loss = loss_dict['sop loss']
                    loss = lm_loss + sop_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss, sop_loss])
                else:
                    lm_loss = loss_dict['lm loss']
                    loss = lm_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss])
                
                return loss, {'loss': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def loss_func(self, output_tensor):
        queries = output_tensor[0]
        positives = output_tensor[1]

        pos_inbatch_negs_scores = torch.mm(queries, positives.transpose(0, 1))

        hard_negs = output_tensor[2:]

        hard_negs_scores = (
            torch.multiply(
                queries.unsqueeze(0).repeat(len(hard_negs), 1, 1),
                torch.stack(hard_negs),
            )
            .sum(axis=-1)
            .T
        )

        scores = torch.cat([pos_inbatch_negs_scores, hard_negs_scores], axis=1)

        scores *= self.scale

        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Indices of the (query, positive) pairs

        return {'lm loss': self.cross_entropy_loss(scores, labels)}

