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

import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset,
    BertPunctuationCapitalizationInferDataset,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging

__all__ = ['PunctuationCapitalizationModel']


class PunctuationCapitalizationModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_logits": NeuralType(('B', 'T', 'C'), LogitsType()),
            "capit_logits": NeuralType(('B', 'T', 'C'), LogitsType()),
        }

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initializes BERT Punctuation and Capitalization model.
        """
        self._setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        self.punct_classifier = TokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_classes=len(self._cfg.punct_label_ids),
            activation=cfg.punct_head.activation,
            log_softmax=cfg.punct_head.log_softmax,
            dropout=cfg.punct_head.fc_dropout,
            num_layers=cfg.punct_head.punct_num_fc_layers,
            use_transformer_init=cfg.punct_head.use_transformer_init,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_classes=len(self._cfg.capit_label_ids),
            activation=cfg.capit_head.activation,
            log_softmax=cfg.capit_head.log_softmax,
            dropout=cfg.capit_head.fc_dropout,
            num_layers=cfg.capit_head.capit_num_fc_layers,
            use_transformer_init=cfg.capit_head.use_transformer_init,
        )

        self.loss = CrossEntropyLoss(logits_ndim=3)
        self.agg_loss = AggregatorLoss(num_inputs=2)

        # setup to track metrics
        self.punct_class_report = ClassificationReport(
            num_classes=len(self._cfg.punct_label_ids), label_ids=self._cfg.punct_label_ids, mode='macro'
        )
        self.capit_class_report = ClassificationReport(
            num_classes=len(self._cfg.capit_label_ids), label_ids=self._cfg.capit_label_ids, mode='macro'
        )

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        capit_logits = self.capit_classifier(hidden_states=hidden_states)
        return punct_logits, capit_logits

    def _make_step(self, batch):
        input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, punct_labels, capit_labels = batch
        punct_logits, capit_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        punct_loss = self.loss(logits=punct_logits, labels=punct_labels, loss_mask=loss_mask)
        capit_loss = self.loss(logits=capit_logits, labels=capit_labels, loss_mask=loss_mask)
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        return loss, punct_logits, capit_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        loss, _, _ = self._make_step(batch)
        tensorboard_logs = {'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        _, _, _, subtokens_mask, _, punct_labels, capit_labels = batch
        val_loss, punct_logits, capit_logits = self._make_step(batch)

        subtokens_mask = subtokens_mask > 0.5
        punct_preds = torch.argmax(punct_logits, axis=-1)[subtokens_mask]
        punct_labels = punct_labels[subtokens_mask]
        self.punct_class_report(punct_preds, punct_labels)

        capit_preds = torch.argmax(capit_logits, axis=-1)[subtokens_mask]
        capit_labels = capit_labels[subtokens_mask]
        self.capit_class_report(capit_preds, capit_labels)
        tensorboard_logs = {
            'val_loss': val_loss,
            'punct_tp': self.punct_class_report.tp,
            'punct_fn': self.punct_class_report.fn,
            'punct_fp': self.punct_class_report.fp,
            'capit_tp': self.capit_class_report.tp,
            'capit_fn': self.capit_class_report.fn,
            'capit_fp': self.capit_class_report.fp,
        }

        return {
            'val_loss': val_loss,
            'log': tensorboard_logs,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1 = self.punct_class_report.compute()

        # calculate metrics and log classification report for Capitalization task
        capit_precision, capit_recall, capit_f1 = self.capit_class_report.compute()

        tensorboard_logs = {
            'validation_loss': avg_loss,
            'punct_precision': punct_precision,
            'punct_f1': punct_f1,
            'punct_recall': punct_recall,
            'capit_precision': capit_precision,
            'capit_f1': capit_f1,
            'capit_recall': capit_recall,
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            vocab_file=self.register_artifact(config_path='tokenizer.vocab_file', src=cfg.vocab_file),
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            tokenizer_model=self.register_artifact(config_path='tokenizer.tokenizer_model', src=cfg.tokenizer_model),
        )
        self.tokenizer = tokenizer

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None, data_dir=None):
        if train_data_config is None:
            train_data_config = self._cfg.train_ds
        if data_dir:
            self._cfg.dataset.data_dir = data_dir
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.register_artifact('punct_label_ids.csv', self._train_dl.dataset.punct_label_ids_file)
            self.register_artifact('capit_label_ids.csv', self._train_dl.dataset.capit_label_ids_file)

            # save label maps to the config
            self._cfg.punct_label_ids = OmegaConf.create(self._train_dl.dataset.punct_label_ids)
            self._cfg.capit_label_ids = OmegaConf.create(self._train_dl.dataset.capit_label_ids)

    def setup_validation_data(
        self, val_data_config: Optional[Dict] = None, data_dirs: Optional[Union[List[str], str]] = None
    ):
        """
        Setup validaton data

        val_data_config: validation data config
        data_dirs: path or paths to validation data dirs, used when setup up data for pretrained model
        """
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds
        if data_dirs:
            self._cfg.validation_ds.ds_item = data_dirs
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Dict] = None):
        if test_data_config is None:
            test_data_config = self._cfg.test_ds
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        # use data_dir specified in the ds_item to run evaluation on multiple datasets
        if 'ds_item' in cfg and cfg.ds_item is not None:
            data_dir = cfg.ds_item
        else:
            data_dir = self._cfg.dataset.data_dir

        text_file = os.path.join(data_dir, cfg.text_file)
        label_file = os.path.join(data_dir, cfg.labels_file)

        dataset = BertPunctuationCapitalizationDataset(
            tokenizer=self.tokenizer,
            text_file=text_file,
            label_file=label_file,
            pad_label=self._cfg.dataset.pad_label,
            punct_label_ids=self._cfg.punct_label_ids,
            capit_label_ids=self._cfg.capit_label_ids,
            max_seq_length=self._cfg.dataset.max_seq_length,
            ignore_extra_tokens=self._cfg.dataset.ignore_extra_tokens,
            ignore_start_end=self._cfg.dataset.ignore_start_end,
            use_cache=self._cfg.dataset.use_cache,
            num_samples=cfg.num_samples,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=self._cfg.dataset.drop_last,
        )

    def _setup_infer_dataloader(self, queries: List[str], batch_size: int) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.

        Args:
            queries: lower cased text without punctuation
            batch_size: batch size to use during inference

        Returns:
            A pytorch DataLoader.
        """

        dataset = BertPunctuationCapitalizationInferDataset(
            tokenizer=self.tokenizer, queries=queries, max_seq_length=self._cfg.dataset.max_seq_length
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=False,
        )

    def add_punctuation_capitalization(self, queries: List[str], batch_size: int = None) -> List[str]:
        """
        Adds punctuation and capitalization to the queries. Use this method for debugging and prototyping.
        Args:
            queries: lower cased text without punctuation
            batch_size: batch size to use during inference
        Returns:
            result: text with added capitalization and punctuation
        """
        if queries is None or len(queries) == 0:
            return []
        if batch_size is None:
            batch_size = len(queries)
            logging.info(f'Using batch size {batch_size} for inference')

        # We will store the output here
        result = []

        # Model's mode and device
        mode = self.training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # Switch model to evaluation mode
            self.eval()
            self = self.to(device)
            infer_datalayer = self._setup_infer_dataloader(queries, batch_size)

            # store predictions for all queries in a single list
            all_punct_preds = []
            all_capit_preds = []

            for batch in infer_datalayer:
                input_ids, input_type_ids, input_mask, subtokens_mask = batch

                punct_logits, capit_logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )

                subtokens_mask = subtokens_mask > 0.5
                punct_preds = tensor2list(torch.argmax(punct_logits, axis=-1)[subtokens_mask])
                capit_preds = tensor2list(torch.argmax(capit_logits, axis=-1)[subtokens_mask])
                all_punct_preds.extend(punct_preds)
                all_capit_preds.extend(capit_preds)

            queries = [q.strip().split() for q in queries]
            queries_len = [len(q) for q in queries]

            if sum(queries_len) != len(all_punct_preds) or sum(queries_len) != len(all_capit_preds):
                raise ValueError('Pred and words must have the same length')

            punct_ids_to_labels = {v: k for k, v in self._cfg.punct_label_ids.items()}
            capit_ids_to_labels = {v: k for k, v in self._cfg.capit_label_ids.items()}

            start_idx = 0
            end_idx = 0
            for query in queries:
                end_idx += len(query)

                # extract predictions for the current query from the list of all predictions
                punct_preds = all_punct_preds[start_idx:end_idx]
                capit_preds = all_capit_preds[start_idx:end_idx]
                start_idx = end_idx

                query_with_punct_and_capit = ''
                for j, word in enumerate(query):
                    punct_label = punct_ids_to_labels[punct_preds[j]]
                    capit_label = capit_ids_to_labels[capit_preds[j]]

                    if capit_label != self._cfg.dataset.pad_label:
                        word = word.capitalize()
                    query_with_punct_and_capit += word
                    if punct_label != self._cfg.dataset.pad_label:
                        query_with_punct_and_capit += punct_label
                    query_with_punct_and_capit += ' '

                result.append(query_with_punct_and_capit.strip())
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return result

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="Punctuation_Capitalization_with_BERT",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemonlpmodels/versions/1.0.0a5/files/Punctuation_Capitalization_with_BERT.nemo",
                description="The model was trained with NeMo BERT base uncased checkpoint on a subset of data from the following sources: Tatoeba sentences, books from Project Gutenberg, Fisher transcripts.",
            )
        )
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="Punctuation_Capitalization_with_DistilBERT",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemonlpmodels/versions/1.0.0a5/files/Punctuation_Capitalization_with_DistilBERT.nemo",
                description="The model was trained with DiltilBERT base uncased checkpoint from HuggingFace on a subset of data from the following sources: Tatoeba sentences, books from Project Gutenberg, Fisher transcripts.",
            )
        )
        return result
