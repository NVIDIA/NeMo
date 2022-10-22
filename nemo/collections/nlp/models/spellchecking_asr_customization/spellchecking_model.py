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


import pdb
from time import perf_counter
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import BertConfig, BertModel

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.spellchecking_asr_customization import (
    SpellcheckingAsrCustomizationDataset,
    SpellcheckingAsrCustomizationTestDataset,
    bert_example,
)
from nemo.collections.nlp.data.text_normalization_as_tagging.utils import read_label_map
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.token_classifier import TokenClassifier
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = ["SpellcheckingAsrCustomizationModel"]


@experimental
class SpellcheckingAsrCustomizationModel(NLPModel):
    """
    BERT-based model for Spellchecking ASR Customization.
    It takes as input ASR hypothesis and candidate customization entries.
    It labels the hypothesis with correct entry index or 0.
    Example input: [CLS] c a l l _ j o h n [SEP] j o n [SEP] 
    Example output:    0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(('B', 'T', 'D'), LogitsType()),
        }

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)

        label_map_file = self.register_artifact("label_map", cfg.label_map, verify_src_exists=True)
        semiotic_classes_file = self.register_artifact(
            "semiotic_classes", cfg.semiotic_classes, verify_src_exists=True
        )
        self.label_map = read_label_map(label_map_file)
        self.semiotic_classes = read_label_map(semiotic_classes_file)

        self.num_labels = len(self.label_map)
        self.num_semiotic_labels = len(self.semiotic_classes)
        self.id_2_tag = {tag_id: tag for tag, tag_id in self.label_map.items()}
        self.id_2_semiotic = {semiotic_id: semiotic for semiotic, semiotic_id in self.semiotic_classes.items()}
        self.max_sequence_len = cfg.get('max_sequence_len', self.tokenizer.tokenizer.model_max_length)

        # setup to track metrics
        # we will have (len(self.semiotic_classes) + 1) labels
        # last one stands for WRONG (span in which the predicted tags don't match the labels)
        # this is needed to feed the sequence of classes to classification_report during validation
        label_ids = self.semiotic_classes.copy()
        label_ids["WRONG"] = len(self.semiotic_classes)
        self.tag_classification_report = ClassificationReport(
            len(self.semiotic_classes) + 1, label_ids=label_ids, mode='micro', dist_sync_on_step=True
        )

        self.hidden_size = cfg.hidden_size

        self.logits = TokenClassifier(
            self.hidden_size, num_classes=self.num_labels, num_layers=1, log_softmax=False, dropout=0.1
        )

        self.loss_fn = CrossEntropyLoss(logits_ndim=3)

        self.builder = bert_example.BertExampleBuilder(
            self.label_map, self.semiotic_classes, self.tokenizer.tokenizer, self.max_sequence_len
        )
        # configuration = BertConfig(type_vocab_size=4)
        # print(configuration)
        # self.bert_model = BertModel(configuration)

    @typecheck()
    def forward(self, input_ids, input_mask, segment_ids):
        src_hiddens = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        logits = self.logits(hidden_states=src_hiddens)
        return logits

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """

        input_ids, input_mask, segment_ids, labels_mask, labels, _ = batch
        logits = self.forward(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        loss = self.loss_fn(logits=logits, labels=labels, loss_mask=labels_mask)
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
        input_ids, input_mask, segment_ids, labels_mask, labels, spans = batch
        logits = self.forward(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        tag_preds = torch.argmax(logits, dim=2)

        # Update tag classification_report
        predictions, tag_labels = tag_preds.tolist(), labels.tolist()
        # if self.current_epoch == 3:
        #    pdb.set_trace()
        for prediction, label, span in zip(predictions, tag_labels, spans):
            # Here we want to track whether the predicted output matches ground truth labels
            # for each whole span
            # so we construct the special input for classification report, for example:
            #   label = [PLAIN, PLAIN, CUSTOM, PLAIN, PLAIN]
            #   pred = [PLAIN, PLAIN, WRONG, PLAIN, PLAIN]
            span_labels = []
            span_predictions = []
            for cid, start, end in span:
                if cid == -1:
                    break
                span_labels.append(cid)
                if prediction[start:end] == label[start:end]:
                    span_predictions.append(cid)
                else:
                    span_predictions.append(self.tag_classification_report.num_classes - 1)  # this stands for WRONG
            if len(span_labels) != len(span_predictions):
                raise ValueError(
                    "Length mismatch: len(span_labels)="
                    + str(len(span_labels))
                    + "; len(span_predictions)="
                    + str(len(span_predictions))
                )
            self.tag_classification_report(
                torch.tensor(span_predictions).to(self.device), torch.tensor(span_labels).to(self.device)
            )

        val_loss = self.loss_fn(logits=logits, labels=labels, loss_mask=labels_mask)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and classification report
        # In our task recall = accuracy, and the recall column - is the per class accuracy
        _, tag_accuracy, _, tag_report = self.tag_classification_report.compute()

        logging.info("Total tag accuracy: " + str(tag_accuracy))
        logging.info(tag_report)

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('tag accuracy', tag_accuracy)

        self.tag_classification_report.reset()

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
            sents: A list of tab-separated text records. Each record consists of 2 items:
                - ASR hypothesis
                - candidate customization entry

        Returns:
            all_preds: A list of tab-separated text records, same size as input list. Each record consists of 4 items:
                - final output text
                - ASR hypothesis
                - labels
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
            hyp, ref = sent.split("\t")
            example = self.builder.build_bert_example(hyp=hyp, ref=ref, infer=True)
            tag_preds = tensor2list(torch.argmax(tag_logits[i], dim=-1))

            # this mask is required by get_token_labels
            example.features["labels_mask"] = [0] + [1] * (len(tag_preds) - 2) + [0]
            example.features["labels"] = tag_preds
            tags = [self.id_2_tag[label_id] for label_id in example.get_token_labels("labels")]
            letters = hyp.split(" ")
            candidates = ref.split(";")
            report_str = ""
            tag_preds_for_sent = tag_preds[1 : (len(letters) + 1)]  # take only predictions for actual sent letters
            last_tag = -1
            tag_begin = -1
            for idx, tag in enumerate(tag_preds_for_sent):
                if tag != last_tag:
                    if last_tag >= 1 and (tag_begin == 0 or letters[tag_begin - 1] == '_') and letters[idx] == "_":
                        source = " ".join(letters[tag_begin:idx])
                        source_len = source.count(" ") + 1
                        target = candidates[last_tag - 1]
                        target_len = target.count(" ") + 1
                        len_ratio = target_len / source_len
                        if len_ratio >= 0.7 and len_ratio <= 1.3:
                            report_str += "REPLACE:\t" + source + "\t" + target + "\t" + hyp + "\n"
                    tag_begin = idx
                last_tag = tag
            if last_tag >= 1 and (tag_begin == 0 or letters[tag_begin - 1] == '_'):
                source = " ".join(letters[tag_begin:])
                source_len = source.count(" ") + 1
                target = candidates[last_tag - 1]
                target_len = target.count(" ") + 1
                len_ratio = target_len / source_len
                if len_ratio >= 0.7 and len_ratio <= 1.3:
                    report_str += "REPLACE:\t" + source + "\t" + target + "\t" + hyp + "\n"

            # pdb.set_trace()
            all_preds.append(
                "\n"
                + " ".join(letters)
                + "\n"
                + " ".join(list(map(str, tag_preds_for_sent)))
                + "\n\t"
                + ref
                + "\n"
                + report_str
            )

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
        dataset = SpellcheckingAsrCustomizationDataset(input_file=input_file, example_builder=self.builder)
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
        Returns:
            A pytorch DataLoader.
        """
        dataset = SpellcheckingAsrCustomizationTestDataset(sents=queries, example_builder=self.builder)
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
        result = []
        return None
