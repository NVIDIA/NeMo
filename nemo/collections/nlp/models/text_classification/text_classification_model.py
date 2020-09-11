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
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from nemo.utils import logging

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_classification import TextClassificationDataDesc, TextClassificationDataset
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.collections.nlp.parts.utils_funcs import get_classification_report, plot_confusion_matrix, tensor2list


__all__ = ['TextClassificationModel']


class TextClassificationModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes the BERTTextClassifier model.
        """

        # shared params for dataset and data loaders
        self.dataset_cfg = cfg.dataset

        self._setup_tokenizer(cfg.tokenizer)
        # init superclass
        super().__init__(cfg=cfg, trainer=trainer)

        self.data_desc = TextClassificationDataDesc(
            train_file=cfg.train_ds.file_path, val_files=[cfg.validation_ds.file_path]
        )

        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        self.classifier = SequenceClassifier(
            hidden_size=self.bert_model.hidden_size,
            num_classes=self.data_desc.num_classes,
            num_layers=cfg.classifier_head.num_output_layers,
            activation='relu',
            log_softmax=False,
            dropout=cfg.classifier_head.fc_dropout,
            use_transformer_init=True,
            idx_conditioned_on=0,
        )

        if cfg.dataset.class_balancing == 'weighted_loss':
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.loss = CrossEntropyLoss(weight=self.data_desc.class_weights)
        else:
            self.loss = CrossEntropyLoss()

        # setup to track metrics
        self.classification_report = ClassificationReport(self.data_desc.num_classes)

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        logits = self.classifier(hidden_states=hidden_states)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        train_loss = self.loss(logits=logits, labels=labels)

        tensorboard_logs = {'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

        preds = torch.argmax(logits, axis=-1)
        tp, fp, fn = self.classification_report(preds, labels)

        tensorboard_logs = {'val_loss': val_loss, 'val_tp': tp, 'val_fn': fn, 'val_fp': fp}

        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return {}

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # calculate metrics and log classification report
        tp = torch.sum(torch.stack([x['log']['val_tp'] for x in outputs]), 0)
        fn = torch.sum(torch.stack([x['log']['val_fn'] for x in outputs]), 0)
        fp = torch.sum(torch.stack([x['log']['val_fp'] for x in outputs]), 0)
        precision, recall, f1 = self.classification_report.get_precision_recall_f1(tp, fn, fp, mode='micro')

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(
            tokenizer_name=cfg.tokenizer_name,
            tokenizer_model=cfg.tokenizer_model,
            special_tokens=OmegaConf.to_container(cfg.special_tokens) if cfg.special_tokens else None,
            vocab_file=cfg.vocab_file,
        )
        self.tokenizer = tokenizer

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file_path:
            logging.info("Dataloader config or file_path for the train is missing, so no data loader for train is created!")
            self._train_dl = None
        else:
            self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file_path:
            logging.info("Dataloader config or file_path for the validation is missing, so no data loader for validation is created!")
            self._validation_dl = None
        else:
            self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)


    def _setup_dataloader_from_config(self, cfg: DictConfig):
        input_file = cfg.file_path
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f'{input_file} not found! The data should be be stored in TAB-separated files \n\
                "validation_ds.file_path" and "train_ds.file_path" for train and evaluation respectively. \n\
                Each line of the files contains text sequences, where words are separated with spaces. \n\
                The label of the example is separated with TAB at the end of each line. \n\
                Each line of the files should follow the format: \n\
                [WORD][SPACE][WORD][SPACE][WORD][...][TAB][LABEL]'
            )

        dataset = TextClassificationDataset(
            tokenizer=self.tokenizer,
            input_file=input_file,
            max_seq_length=self.dataset_cfg.max_seq_length,
            num_samples=cfg.num_samples,
            shuffle=cfg.shuffle,
            use_cache=self.dataset_cfg.use_cache,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=self.dataset_cfg.num_workers,
            pin_memory=self.dataset_cfg.pin_memory,
            drop_last=self.dataset_cfg.drop_last,
            collate_fn=dataset.collate_fn,
        )

    def infer(self, queries: List[str], batch_size: int = None, device=None) -> List[int]:
        """
        Get prediction for the queries
        Args:
            queries: text sequences
            batch_size: batch size to use during inference.
            device: device to perform the inference on, eg. cuda or cpu
        Returns:
            all_preds: model predictions
        """
        if device is None:
            device = self.device
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        try:
            # Switch model to evaluation mode
            self.eval()
            self.to(device)
            infer_datalayer = self._setup_infer_dataloader(queries, batch_size)

            for i, batch in enumerate(infer_datalayer):
                input_ids, input_type_ids, input_mask, subtokens_mask = batch

                logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )

                preds = tensor2list(torch.argmax(logits, axis=-1))
                all_preds.extend(preds)
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return all_preds

    def _setup_infer_dataloader(self, queries: List[str], batch_size: int, max_seq_length: int = -1) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.

        Args:
            queries: text
            batch_size: batch size to use during inference
            max_seq_length: maximum length of queries, default is -1 for no limit
        Returns:
            A pytorch DataLoader.
        """
        dataset = TextClassificationDataset(tokenizer=self.tokenizer, queries=queries, max_seq_length=max_seq_length)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass
