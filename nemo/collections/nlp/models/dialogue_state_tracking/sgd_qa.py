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

import onnx
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.utils.export_utils import attach_onnx_to_onnx

__all__ = ['SGDQA']


class SGDQA(NLPModel, Exportable):
    """Dialogue State Tracking Model SGD-QA"""

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        schema_config = {
            "MAX_NUM_CAT_SLOT": 6,
            "MAX_NUM_NONCAT_SLOT": 12,
            "MAX_NUM_VALUE_PER_CAT_SLOT": 12,
            "MAX_NUM_INTENT": 4,
            "NUM_TASKS": 6,
            "MAX_SEQ_LENGTH": self._cfg.dataset.max_sequence_length
        }
        self.setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )


        all_schema_json_paths = []
        for dataset_split in ['train', 'test', 'dev']:
            all_schema_json_paths.append(os.path.join(args.data_dir, dataset_split, "schema.json"))
        schemas = schema.Schema(all_schema_json_paths)

        dialogues_processor = data_processor.SGDDataProcessor(
            task_name=self._cfg.dataset.task_name,
            data_dir=self._cfg.dataset.data_dir,
            dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
            tokenizer=self.tokenizer,
            schemas=schemas,
            schema_config=schema_config,
            subsample=self._cfg.dataset.subsample,
            overwrite_dial_files=self._cfg.dataset.overwrite_dial_files,
        )

        self.encoder = SGDEncoder(hidden_size=self.bert_model.config.hidden_size, dropout=self._cfg.encoder.dropout)
        self.decoder = SGDDecoder(embedding_dim=self.bert_model.config.hidden_size)
        self.loss = SGDDialogueStateLoss(reduction="mean")

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
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
        input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss = self.loss(logits=logits, labels=labels, loss_mask=loss_mask)
        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        val_loss = self.loss(logits=logits, labels=labels, loss_mask=loss_mask)

        subtokens_mask = subtokens_mask > 0.5

        preds = torch.argmax(logits, axis=-1)[subtokens_mask]
        labels = labels[subtokens_mask]
        tp, fn, fp, _ = self.classification_report(preds, labels)

        return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and classification report
        precision, recall, f1, report = self.classification_report.compute()

        logging.info(report)

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('precision', precision)
        self.log('f1', f1)
        self.log('recall', recall)

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        if train_data_config is None:
            train_data_config = self._cfg.train_ds

        labels_file = os.path.join(self._cfg.dataset.data_dir, train_data_config.labels_file)

        # for older(pre - 1.0.0.b3) configs compatibility
        if not hasattr(self._cfg, "class_labels") or self._cfg.class_labels is None:
            OmegaConf.set_struct(self._cfg, False)
            self._cfg.class_labels = {}
            self._cfg.class_labels = OmegaConf.create({'class_labels_file': 'label_ids.csv'})
            OmegaConf.set_struct(self._cfg, True)

        label_ids, label_ids_filename, self.class_weights = get_label_ids(
            label_file=labels_file,
            is_training=True,
            pad_label=self._cfg.dataset.pad_label,
            label_ids_dict=self._cfg.label_ids,
            get_weights=True,
            class_labels_file_artifact=self._cfg.class_labels.class_labels_file,
        )
        # save label maps to the config
        self._cfg.label_ids = OmegaConf.create(label_ids)

        self.register_artifact(self._cfg.class_labels.class_labels_file, label_ids_filename)
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds

        labels_file = os.path.join(self._cfg.dataset.data_dir, val_data_config.labels_file)
        get_label_ids(
            label_file=labels_file,
            is_training=False,
            pad_label=self._cfg.dataset.pad_label,
            label_ids_dict=self._cfg.label_ids,
            get_weights=False,
        )

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    
    def _setup_dataloader_from_config(self, cfg: DictConfig) -> DataLoader:
        """
        Setup dataloader from config
        Args:
            cfg: config for the dataloader
        Return:
            Pytorch Dataloader
        """
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        text_file = os.path.join(data_dir, cfg.text_file)
        labels_file = os.path.join(data_dir, cfg.labels_file)

        if not (os.path.exists(text_file) and os.path.exists(labels_file)):
            raise FileNotFoundError(
                f'{text_file} or {labels_file} not found. The data should be split into 2 files: text.txt and \
                labels.txt. Each line of the text.txt file contains text sequences, where words are separated with \
                spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are \
                separated with spaces. Each line of the files should follow the format:  \
                   [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
                   [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )
        dataset = BertTokenClassificationDataset(
            text_file=text_file,
            label_file=labels_file,
            max_seq_length=dataset_cfg.max_seq_length,
            tokenizer=self.tokenizer,
            num_samples=cfg.num_samples,
            pad_label=dataset_cfg.pad_label,
            label_ids=self._cfg.label_ids,
            ignore_extra_tokens=dataset_cfg.ignore_extra_tokens,
            ignore_start_end=dataset_cfg.ignore_start_end,
            use_cache=dataset_cfg.use_cache,
        )
        return DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=dataset_cfg.num_workers,
            pin_memory=dataset_cfg.pin_memory,
            drop_last=dataset_cfg.drop_last,
        )



    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        pass

  