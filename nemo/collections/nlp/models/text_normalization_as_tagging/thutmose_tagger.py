# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from time import perf_counter
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_normalization_as_tagging import (
    ThutmoseTaggerDataset,
    ThutmoseTaggerTestDataset,
    bert_example,
    tagging,
)
from nemo.collections.nlp.data.text_normalization_as_tagging.utils import read_label_map, read_semiotic_classes
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.token_classifier import TokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import ChannelType, LogitsType, MaskType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ['ThutmoseTaggerModel']


@experimental
class ThutmoseTaggerModel(NLPModel):
    """
    BERT-based tagging model for ITN, inspired by LaserTagger approach.
    It maps spoken-domain input words to tags:
        KEEP, DELETE, or any of predefined replacement tags which correspond to a written-domain fragment.
    Example: one hundred thirty four -> _1 <DELETE> 3 4_
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), MaskType()),
            "segment_ids": NeuralType(('B', 'T'), MaskType()),
            # "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'T', 'D'), LogitsType())}

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)

        label_map_file = self.register_artifact("label_map", cfg.label_map)
        semiotic_classes_file = self.register_artifact("semiotic_classes", cfg.semiotic_classes)
        self.label_map = read_label_map(label_map_file)
        self.semiotic_classes = read_semiotic_classes(semiotic_classes_file)

        self.num_labels = len(self.label_map)
        self.id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in self.label_map.items()}
        self.max_sequence_len = cfg.get('max_sequence_len', self.tokenizer.tokenizer.model_max_length)

        # setup to track metrics
        # we will have (len(self.semiotic_classes) + 1) labels, last one stands for WRONG
        label_ids = self.semiotic_classes.copy()
        label_ids["WRONG"] = len(self.semiotic_classes)
        self.classification_report = ClassificationReport(
            len(self.semiotic_classes) + 1, label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )
        self.multiword_classification_report = ClassificationReport(
            len(self.semiotic_classes) + 1, label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )

        self.hidden_size = cfg.hidden_size

        self.logits = TokenClassifier(
            self.hidden_size, num_classes=self.num_labels, num_layers=1, log_softmax=False, dropout=0.1
        )

        self.loss_fn = CrossEntropyLoss(logits_ndim=3)
        self.loss_eval_metric = CrossEntropyLoss(logits_ndim=3, reduction='none')

        converter = tagging.TaggingConverterTrivial()

        self.builder = bert_example.BertExampleBuilder(
            self.label_map, self.semiotic_classes, self.tokenizer.tokenizer, self.max_sequence_len, converter
        )

    @typecheck()
    def forward(self, input_ids, input_mask, segment_ids):

        src_hiddens = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        log_softmax = self.logits(hidden_states=src_hiddens)
        return log_softmax

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """

        input_ids, input_mask, segment_ids, labels_mask, labels, _ = batch
        tag_logits = self.forward(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        loss = self.loss_fn(logits=tag_logits, labels=labels, loss_mask=labels_mask)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': loss, 'lr': lr}

    # Validation and Testing
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_mask, segment_ids, labels_mask, batch_labels, semiotic_classes = batch
        tag_logits = self.forward(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        tag_preds = torch.argmax(tag_logits, dim=2)
        # Update classification_report
        predictions, labels = tag_preds.tolist(), batch_labels.tolist()
        for prediction, label, semiotic in zip(predictions, labels, semiotic_classes):
            # Here we want to track whether the predicted output matches ground truth labels for each whole semiotic span
            # so we construct the special input for classification report, for example:
            #   label = [PLAIN, PLAIN, DATE, PLAIN, LETTERS, PLAIN]
            #   pred = [PLAIN, PLAIN, WRONG, PLAIN, LETTERS, PLAIN]
            semiotic_labels = []
            semiotic_predictions = []
            for cid, start, end in semiotic:
                if cid == -1:
                    break
                semiotic_labels.append(cid)
                if prediction[start:end] == label[start:end]:
                    semiotic_predictions.append(cid)
                else:
                    semiotic_predictions.append(self.classification_report.num_classes - 1)  # this stands for WRONG
            assert len(semiotic_labels) == len(semiotic_predictions)
            self.classification_report(
                torch.tensor(semiotic_predictions).to(self.device), torch.tensor(semiotic_labels).to(self.device)
            )

            multiword_semiotic_labels = []
            multiword_semiotic_predictions = []
            for cid, start, end in semiotic:
                if cid == -1:
                    break
                # this is a trick to determine if label consists of a single replacement
                # - it will be repeated for each input subtoken
                if len(set(label[start:end])) == 1:
                    continue
                multiword_semiotic_labels.append(cid)
                if prediction[start:end] == label[start:end]:
                    multiword_semiotic_predictions.append(cid)
                else:
                    # this stands for WRONG
                    multiword_semiotic_predictions.append(self.classification_report.num_classes - 1)
            assert len(multiword_semiotic_labels) == len(multiword_semiotic_predictions)
            self.multiword_classification_report(
                torch.tensor(multiword_semiotic_predictions).to(self.device),
                torch.tensor(multiword_semiotic_labels).to(self.device),
            )

        val_loss = self.loss_fn(logits=tag_logits, labels=batch_labels, loss_mask=labels_mask)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and classification report
        _, accuracy, _, report = self.classification_report.compute()
        _, multiword_accuracy, _, multiword_report = self.multiword_classification_report.compute()

        logging.info("Total accuracy")
        logging.info(report)
        logging.info("Only multiword accuracy")
        logging.info(multiword_report)

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('accuracy', accuracy)
        self.log('multiword_accuracy', multiword_accuracy)

        self.classification_report.reset()
        self.multiword_classification_report.reset()

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.validation_epoch_end(outputs)

    # Functions for inference
    @torch.no_grad()
    def _infer(self, sents: List[str]) -> List[List[int]]:
        """ Main function for Inference

        Args:
            sents: A list of input sentences (lowercase spoken-domain words separated by space).

        Returns:
            all_preds: A list of tab-separated text records, same size as input list. Each record consists of 4 items:
                - final output text
                - input words
                - tags predicted for input words
                - tags after swap preprocessing
        """

        # all input sentences go into one batch
        dataloader_cfg = {"batch_size": len(sents), "num_workers": 3, "pin_memory": False}
        infer_datalayer = self._setup_infer_dataloader(dataloader_cfg, sents)

        batch = next(iter(infer_datalayer))
        input_ids, input_mask, segment_ids = batch

        tag_logits = self.forward(
            input_ids=input_ids.to(self.device),
            input_mask=input_mask.to(self.device),
            segment_ids=segment_ids.to(self.device),
        )

        all_preds = []
        for i, sent in enumerate(sents):
            example = self.builder.build_bert_example(source=sent, infer=True)
            tag_preds = tensor2list(torch.argmax(tag_logits[i], dim=-1))
            example.features['labels'] = tag_preds
            # this mask is required by get_token_labels
            example.features['labels_mask'] = [0] + [1] * (len(tag_preds) - 2) + [0]
            labels = [self.id_2_tag[label_id] for label_id in example.get_token_labels()]
            prediction, inp_str, tag_str, tag_with_swap_str = example.editing_task.realize_output(labels)
            all_preds.append(prediction + "\t" + inp_str + "\t" + tag_str + "\t" + tag_with_swap_str)

        return all_preds

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, data_split="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, data_split="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.data_path is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, data_split="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, data_split: str):
        start_time = perf_counter()
        logging.info(f'Creating {data_split} dataset')
        input_file = cfg.data_path
        dataset = ThutmoseTaggerDataset(input_file=input_file, example_builder=self.builder)
        dl = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, collate_fn=dataset.collate_fn
        )
        running_time = perf_counter() - start_time
        logging.info(f'Took {running_time} seconds')
        return dl

    def _setup_infer_dataloader(self, cfg: DictConfig, queries: List[str]) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Args:
            cfg: config dictionary containing data loader params like batch_size, num_workers and pin_memory
            queries: text
            max_seq_length: maximum length of queries, default is -1 for no limit
        Returns:
            A pytorch DataLoader.
        """
        dataset = ThutmoseTaggerTestDataset(sents=queries, example_builder=self.builder)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        pass
