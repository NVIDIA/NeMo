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
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss, MSELoss
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import GLUE_TASKS_NUM_LABELS, GLUEDataset
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.collections.nlp.modules.common.sequence_classifier import SequenceClassifier
from nemo.collections.nlp.modules.common.sequence_regresssion import SequenceRegression
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['GLUEModel']

'''
Some transformer of this code were adapted from the HuggingFace library at
https://github.com/huggingface/transformers
Example of running a pretrained BERT model on the 9 GLUE tasks, read more
about GLUE benchmark here: https://gluebenchmark.com
Download the GLUE data by running the script:
https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e

Some of these tasks have a small dataset and training can lead to high variance
in the results between different runs. Below is the median on 5 runs
(with different seeds) for each of the metrics on the dev set of the benchmark
with an uncased BERT base model (the checkpoint bert-base-uncased)
(source https://github.com/huggingface/transformers/tree/master/examples#glue).
Task	Metric	                        Result
CoLA	Matthew's corr	                48.87
SST-2	Accuracy	                    91.74
MRPC	F1/Accuracy	                 90.70/86.27
STS-B	Person/Spearman corr.	     91.39/91.04
QQP	    Accuracy/F1	                 90.79/87.66
MNLI	Matched acc./Mismatched acc. 83.70/84.83
QNLI	Accuracy	                    89.31
RTE	    Accuracy	                    71.43
WNLI	Accuracy	                    43.66

'''


@experimental
class GLUEModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.pooler.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initializes model to use BERT model for GLUE tasks.
        """

        self.data_dir = cfg.data_dir

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                "GLUE datasets not found. Datasets can be "
                "obtained at https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e"
            )

        if cfg.task_name not in cfg.supported_tasks:
            raise ValueError(f'{cfg.task_name} not in supported task. Choose from {cfg.supported_tasks}')
        self.task_name = cfg.task_name
        self.model_cfg = cfg

        self.tokenizer = get_tokenizer(
            tokenizer_name=cfg.language_model.tokenizer,
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            vocab_file=cfg.language_model.vocab_file,
            tokenizer_model=cfg.language_model.tokenizer_model,
            do_lower_case=cfg.language_model.do_lower_case,
        )

        super().__init__(cfg=cfg, trainer=trainer)
        """
        Prepare GLUE task
        MNLI task has two separate dev sets: matched and mismatched
        """
        if cfg.task_name == "mnli":
            cfg.validation_ds.file_name = 'dev_matched.tsv'
            # TODO add eval for mismathced dev set

        num_labels = GLUE_TASKS_NUM_LABELS[self.task_name]

        self.bert_model = get_pretrained_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.bert_config,
            checkpoint_file=cfg.language_model.bert_checkpoint,
        )
        self.hidden_size = self.bert_model.config.hidden_size

        # uses [CLS] token for classification (the first token)
        if self.task_name == "sts-b":
            self.pooler = SequenceRegression(hidden_size=self.hidden_size)
            self.loss = MSELoss()
        else:
            self.pooler = SequenceClassifier(hidden_size=self.hidden_size, num_classes=num_labels, log_softmax=False)
            self.loss = CrossEntropyLoss()

        # # setup to track metrics
        # TODO setup metrics

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        output = self.pooler(hidden_states=hidden_states)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, labels = batch
        model_output = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        if self.task_name == "sts-b":
            loss = self.loss(preds=model_output, labels=labels)
        else:
            loss = self.loss(logits=model_output, labels=labels)
        tensorboard_logs = {'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, labels = batch
        model_output = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        if self.task_name == "sts-b":
            val_loss = self.loss(preds=model_output, labels=labels)
        else:
            val_loss = self.loss(logits=model_output, labels=labels)

        tensorboard_logs = {'val_loss': val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self.__setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = GLUEDataset(
            file_name=os.path.join(self.data_dir, cfg.file_name),
            task_name=self.task_name,
            tokenizer=self.tokenizer,
            max_seq_length=self.model_cfg.max_seq_length,
            use_cache=self.model_cfg.use_cache,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass
