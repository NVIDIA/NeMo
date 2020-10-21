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

import json
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import SpanningLoss
from nemo.collections.nlp.data import SquadDataset
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['QAModel']


class QAModel(NLPModel):
    """
    BERT encoder with QA head training.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        self.classifier = TokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_classes=cfg.token_classifier.num_classes,
            num_layers=cfg.token_classifier.num_layers,
            activation=cfg.token_classifier.activation,
            log_softmax=cfg.token_classifier.log_softmax,
            dropout=cfg.token_classifier.dropout,
            use_transformer_init=cfg.token_classifier.use_transformer_init,
        )

        self.loss = SpanningLoss()

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        logits = self.classifier(hidden_states=hidden_states)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, _, _ = self.loss(logits=logits, start_positions=start_positions, end_positions=end_positions)
        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)

        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, start_logits, end_logits = self.loss(
            logits=logits, start_positions=start_positions, end_positions=end_positions
        )

        eval_tensors = {
            'unique_ids': unique_ids,
            'start_logits': start_logits,
            'end_logits': end_logits,
        }
        self.log('val_loss', loss)
        return {'val_loss': loss, 'eval_tensors': eval_tensors}

    def test_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        test_tensors = {
            'unique_ids': unique_ids,
            'logits': logits,
        }
        return {'test_tensors': test_tensors}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        unique_ids = torch.cat([x['eval_tensors']['unique_ids'] for x in outputs])
        start_logits = torch.cat([x['eval_tensors']['start_logits'] for x in outputs])
        end_logits = torch.cat([x['eval_tensors']['end_logits'] for x in outputs])

        all_unique_ids = []
        all_start_logits = []
        all_end_logits = []
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_unique_ids.append(torch.empty_like(unique_ids))
                all_start_logits.append(torch.empty_like(start_logits))
                all_end_logits.append(torch.empty_like(end_logits))
            torch.distributed.all_gather(all_unique_ids, unique_ids)
            torch.distributed.all_gather(all_start_logits, start_logits)
            torch.distributed.all_gather(all_end_logits, end_logits)
        else:
            all_unique_ids.append(unique_ids)
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)

        exact_match, f1, all_predictions, all_nbest = -1, -1, [], []
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

            unique_ids = []
            start_logits = []
            end_logits = []
            for u in all_unique_ids:
                unique_ids.extend(tensor2list(u))
            for u in all_start_logits:
                start_logits.extend(tensor2list(u))
            for u in all_end_logits:
                end_logits.extend(tensor2list(u))

            exact_match, f1, all_predictions, all_nbest = self.validation_dataset.evaluate(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits,
                n_best_size=self._cfg.validation_ds.n_best_size,
                max_answer_length=self._cfg.validation_ds.max_answer_length,
                version_2_with_negative=self._cfg.dataset.version_2_with_negative,
                null_score_diff_threshold=self._cfg.validation_ds.null_score_diff_threshold,
                do_lower_case=self._cfg.dataset.do_lower_case,
            )

            if self._cfg.validation_ds.output_nbest_file is not None:
                with open(self._cfg.validation_ds.output_nbest_file, "w") as writer:
                    writer.write(json.dumps(all_nbest, indent=4) + "\n")
            if self._cfg.validation_ds.output_prediction_file is not None:
                with open(self._cfg.validation_ds.output_prediction_file, "w") as writer:
                    writer.write(json.dumps(all_predictions, indent=4) + "\n")

        logging.info(f"exact match {exact_match}")
        logging.info(f"f1 {f1}")
        self.log('val_loss', avg_loss)
        self.log('exact_match', exact_match)
        self.log('f1', f1)

    def test_epoch_end(self, outputs):
        unique_ids = tensor2list(torch.cat([x['test_tensors']['unique_ids'] for x in outputs]))
        logits = torch.cat([x['test_tensors']['logits'] for x in outputs])
        s, e = logits.split(dim=-1, split_size=1)
        start_logits = tensor2list(s.squeeze())
        end_logits = tensor2list(e.squeeze())
        (all_predictions, all_nbest, scores_diff) = self.test_dataset.get_predictions(
            unique_ids=unique_ids,
            start_logits=start_logits,
            end_logits=end_logits,
            n_best_size=self._cfg.test_ds.n_best_size,
            max_answer_length=self._cfg.test_ds.max_answer_length,
            version_2_with_negative=self._cfg.dataset.version_2_with_negative,
            null_score_diff_threshold=self._cfg.test_ds.null_score_diff_threshold,
            do_lower_case=self._cfg.dataset.do_lower_case,
        )

        if self._cfg.test_ds.output_nbest_file is not None:
            with open(self._cfg.test_ds.output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest, indent=4) + "\n")
        if self._cfg.test_ds.output_prediction_file is not None:
            with open(self._cfg.test_ds.output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
        return {}

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            tokenizer_model=cfg.tokenizer_model,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            vocab_file=cfg.vocab_file,
        )
        self.tokenizer = tokenizer

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if test_data_config.file is None:
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = SquadDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.file,
            doc_stride=self._cfg.dataset.doc_stride,
            max_query_length=self._cfg.dataset.max_query_length,
            max_seq_length=self._cfg.dataset.max_seq_length,
            version_2_with_negative=self._cfg.dataset.version_2_with_negative,
            num_samples=cfg.num_samples,
            mode=cfg.mode,
            use_cache=self._cfg.dataset.use_cache,
        )
        if cfg.mode == "eval":
            self.validation_dataset = dataset
        elif cfg.mode == "test":
            self.test_dataset = dataset

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get('drop_last', False),
            shuffle=cfg.shuffle,
            num_workers=cfg.get('num_workers', 0),
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="BERTBaseUncasedSQuADv1.1",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemonlpmodels/versions/1.0.0a5/files/BERTBaseUncasedSQuADv1.1.nemo",
            description="Question answering model finetuned from NeMo BERT Base Uncased on SQuAD v1.1 dataset which obtains an exact match (EM) score of 82.43% and an F1 score of 89.59%.",
        )
        result.append(model)
        model = PretrainedModelInfo(
            pretrained_model_name="BERTBaseUncasedSQuADv2.0",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemonlpmodels/versions/1.0.0a5/files/BERTBaseUncasedSQuADv2.0.nemo",
            description="Question answering model finetuned from NeMo BERT Base Uncased on SQuAD v2.0 dataset which obtains an exact match (EM) score of 73.35% and an F1 score of 76.44%.",
        )
        result.append(model)
        model = PretrainedModelInfo(
            pretrained_model_name="BERTLargeUncasedSQuADv1.1",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemonlpmodels/versions/1.0.0a5/files/BERTLargeUncasedSQuADv1.1.nemo",
            description="Question answering model finetuned from NeMo BERT Large Uncased on SQuAD v1.1 dataset which obtains an exact match (EM) score of 85.47% and an F1 score of 92.10%.",
        )
        result.append(model)
        model = PretrainedModelInfo(
            pretrained_model_name="BERTLargeUncasedSQuADv2.0",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemonlpmodels/versions/1.0.0a5/files/BERTLargeUncasedSQuADv2.0.nemo",
            description="Question answering model finetuned from NeMo BERT Large Uncased on SQuAD v2.0 dataset which obtains an exact match (EM) score of 78.8% and an F1 score of 81.85%.",
        )
        result.append(model)
        return result
