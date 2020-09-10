# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.data import BertInformationRetrievalDatasetEval, BertInformationRetrievalDatasetTrain
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import ChannelType, LogitsType, MaskType, NeuralType

__all__ = ['BertDPRModel']


class BertDPRModel(ModelPT):
    """
    Information retrieval model which encodes query and passage separately
    with two different BERT encoders and computes their similarity score
    as a dot-product between corresponding [CLS] token representations.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "q_input_ids": NeuralType(('B', 'T'), ChannelType()),
            "q_attention_mask": NeuralType(('B', 'T'), MaskType()),
            "q_token_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "p_input_ids": NeuralType(('B', 'T'), ChannelType()),
            "p_attention_mask": NeuralType(('B', 'T'), MaskType()),
            "p_token_type_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        self.dataset_cfg = cfg.dataset

        self.tokenizer = get_tokenizer(tokenizer_name=cfg.language_model.pretrained_model_name)

        super().__init__(cfg=cfg, trainer=trainer)

        self.q_encoder = get_lm_model(pretrained_model_name=cfg.language_model.pretrained_model_name)
        self.p_encoder = get_lm_model(pretrained_model_name=cfg.language_model.pretrained_model_name)

        # make vocabulary size divisible by 8 for fast fp16 training
        vocab_size = self.tokenizer.vocab_size
        tokens_to_add = 8 * math.ceil(vocab_size / 8) - vocab_size
        hidden_size = self.q_encoder.embeddings.word_embeddings.weight.shape[1]
        zeros = torch.zeros((tokens_to_add, hidden_size))
        self.q_encoder.embeddings.word_embeddings.weight.data = torch.cat(
            (self.q_encoder.embeddings.word_embeddings.weight.data, zeros)
        )
        self.p_encoder.embeddings.word_embeddings.weight.data = torch.cat(
            (self.p_encoder.embeddings.word_embeddings.weight.data, zeros)
        )

        self.loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(
        self, q_input_ids, q_token_type_ids, q_attention_mask, p_input_ids, p_token_type_ids, p_attention_mask
    ):

        q_vectors = self.q_encoder(
            input_ids=q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attention_mask
        )
        q_vectors = q_vectors[:, 0]
        batch_size, hidden_size = q_vectors.size()

        p_vectors = self.p_encoder(
            input_ids=p_input_ids, token_type_ids=p_token_type_ids, attention_mask=p_attention_mask
        )
        num_passages = p_vectors.shape[0] // batch_size
        p_vectors = p_vectors[:, 0].view(-1, num_passages, hidden_size)
        p_positives, p_negatives = p_vectors[:, 0], p_vectors[:, 1:]
        scores = torch.cat(
            (torch.matmul(q_vectors, p_positives.T), torch.einsum("ij,ipj->ip", q_vectors, p_negatives)), dim=1
        )

        return scores

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        q_input_ids, q_input_mask, q_input_type_ids, p_input_ids, p_input_mask, p_input_type_ids = batch
        batch_size, num_passages, p_seq_length = p_input_ids.size()
        q_seq_length = q_input_ids.size()[-1]

        scores = self(
            q_input_ids=q_input_ids.view(-1, q_seq_length),
            q_token_type_ids=q_input_type_ids.view(-1, q_seq_length),
            q_attention_mask=q_input_mask.view(-1, q_seq_length),
            p_input_ids=p_input_ids.view(-1, p_seq_length),
            p_token_type_ids=p_input_type_ids.view(-1, p_seq_length),
            p_attention_mask=p_input_mask.view(-1, p_seq_length),
        ).view(batch_size, 1, batch_size + num_passages - 1)
        scores = torch.log_softmax(scores, dim=-1)

        labels = torch.arange(batch_size)[:, None].long().to(scores.device)
        train_loss = self.loss(logits=scores, labels=labels, output_mask=torch.ones_like(labels))

        tensorboard_logs = {'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        (
            q_input_ids,
            q_input_mask,
            q_input_type_ids,
            p_input_ids,
            p_input_mask,
            p_input_type_ids,
            query_id,
            passage_ids,
        ) = batch
        batch_size, num_passages, p_seq_length = p_input_ids.size()
        q_seq_length = q_input_ids.size()[-1]

        scores = self(
            q_input_ids=q_input_ids.view(-1, q_seq_length),
            q_token_type_ids=q_input_type_ids.view(-1, q_seq_length),
            q_attention_mask=q_input_mask.view(-1, q_seq_length),
            p_input_ids=p_input_ids.view(-1, p_seq_length),
            p_token_type_ids=p_input_type_ids.view(-1, p_seq_length),
            p_attention_mask=p_input_mask.view(-1, p_seq_length),
        ).view(batch_size, 1, batch_size + num_passages - 1)

        data_for_val = {
            "scores": scores[0, 0, :].detach().cpu().numpy(),
            "query_id": query_id.item(),
            "passage_ids": passage_ids[0].detach().cpu().numpy(),
        }
        return data_for_val

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        query2passages = {}
        for val_output in outputs:
            query_id = val_output["query_id"]

            if query_id in query2passages:
                query2passages[query_id]["psg_ids"] = np.concatenate(
                    (query2passages[query_id]["psg_ids"], val_output["passage_ids"])
                )
                query2passages[query_id]["scores"] = np.concatenate(
                    (query2passages[query_id]["scores"], val_output["scores"])
                )
            else:
                query2passages[query_id] = {"psg_ids": val_output["passage_ids"], "scores": val_output["scores"]}

        rrs = self.calculate_reciprocal_ranks(query2passages, self._val_qrels)

        tensorboard_logs = {
            "model_mrr": np.mean(rrs["model"]),
            "oracle_mrr": np.mean(rrs["oracle"]),
        }

        return {"log": tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode="eval")
        self._val_qrels = self.parse_qrels(val_data_config.qrels)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode="eval")

    @staticmethod
    def parse_qrels(qrels):
        query2rel = {}
        for line in open(qrels, "r").readlines():
            query_id = int(line.split("\t")[0])
            psg_id = int(line.split("\t")[2])
            if query_id not in query2rel:
                query2rel[query_id] = [psg_id]
            else:
                query2rel[query_id].append(psg_id)
        return query2rel

    @staticmethod
    def calculate_reciprocal_ranks(query2passages, query2rel):
        """
        Args:
            query2passages: dict which contains passage ids and corresponding
                scores for each query
            query2rel: dict which contains ids of relevant passages for each query
        """

        oracle_rrs, rrs = [], []

        for query in query2passages:
            indices = np.argsort(query2passages[query]["scores"])[::-1]
            sorted_psgs = query2passages[query]["psg_ids"][indices]

            oracle_rrs.append(0)
            rrs.append(0)
            for i, psg_id in enumerate(sorted_psgs):
                if psg_id in query2rel[query]:
                    rrs[-1] = 1 / (i + 1)
                    oracle_rrs[-1] = 1
                    break
        rrs = {"oracle": oracle_rrs, "model": rrs}
        return rrs

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode="train"):

        dataset_params = {
            "tokenizer": self.tokenizer,
            "passages": cfg.passages,
            "queries": cfg.queries,
            "query_to_passages": cfg.query_to_passages,
            "max_query_length": self.dataset_cfg.get("max_query_length", 31),
            "max_passage_length": self.dataset_cfg.get("max_passage_length", 190),
            "preprocess_fn": "preprocess_dpr",
        }

        if mode == "train":
            dataset = BertInformationRetrievalDatasetTrain(
                num_negatives=cfg.get("num_negatives", 10), **dataset_params,
            )
        elif mode == "eval":
            dataset = BertInformationRetrievalDatasetEval(
                num_candidates=cfg.get("num_candidates", 10), **dataset_params,
            )
        else:
            raise ValueError("mode should be either train or eval")

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=self.dataset_cfg.get("num_workers", 2),
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            drop_last=self.dataset_cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
