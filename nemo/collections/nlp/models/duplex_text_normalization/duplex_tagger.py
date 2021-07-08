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

from time import perf_counter
from typing import List, Optional

import nltk
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization import TextNormalizationTaggerDataset
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.duplex_text_normalization.utils import has_numbers
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging
from nemo.utils.decorators.experimental import experimental

nltk.download('punkt')


__all__ = ['DuplexTaggerModel']


@experimental
class DuplexTaggerModel(NLPModel):
    """
    Transformer-based (duplex) tagger model for TN/ITN.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, add_prefix_space=True)
        super().__init__(cfg=cfg, trainer=trainer)
        self.num_labels = len(constants.ALL_TAG_LABELS)
        self.model = AutoModelForTokenClassification.from_pretrained(cfg.transformer, num_labels=self.num_labels)

        # Loss Functions
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=constants.LABEL_PAD_TOKEN_ID)

        # setup to track metrics
        self.classification_report = ClassificationReport(self.num_labels, mode='micro', dist_sync_on_step=True)

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        num_labels = self.num_labels

        # Apply Transformer
        tag_logits = self.model(batch['input_ids'], batch['attention_mask']).logits

        # Loss
        train_loss = self.loss_fct(tag_logits.view(-1, num_labels), batch['labels'].view(-1))

        lr = self._optimizer.param_groups[0]['lr']
        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)
        return {'loss': train_loss, 'lr': lr}

    # Validation and Testing
    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        # Apply Transformer
        tag_logits = self.model(batch['input_ids'], batch['attention_mask']).logits
        tag_preds = torch.argmax(tag_logits, dim=2)

        # Update classification_report
        predictions, labels = tag_preds.tolist(), batch['labels'].tolist()
        for prediction, label in zip(predictions, labels):
            cur_preds = [p for (p, l) in zip(prediction, label) if l != constants.LABEL_PAD_TOKEN_ID]
            cur_labels = [l for (p, l) in zip(prediction, label) if l != constants.LABEL_PAD_TOKEN_ID]
            self.classification_report(
                torch.tensor(cur_preds).to(self.device), torch.tensor(cur_labels).to(self.device)
            )

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        # calculate metrics and classification report
        precision, _, _, report = self.classification_report.compute()

        logging.info(report)

        self.log('val_token_precision', precision)

        self.classification_report.reset()

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
    def _infer(self, sents: List[List[str]], inst_directions: List[str]):
        """ Main function for Inference
        Args:
            sents: A list of inputs tokenized by a basic tokenizer (e.g., using nltk.word_tokenize()).
            inst_directions: A list of str where each str indicates the direction of the corresponding instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).

        Returns:
            all_tag_preds: A list of list where each list contains the raw tag predictions for the corresponding input.
            nb_spans: A list of ints where each int indicates the number of semiotic spans in each input.
            span_starts: A list of lists where each list contains the starting locations of semiotic spans in an input.
            span_ends: A list of lists where each list contains the ending locations of semiotic spans in an input.
        """
        self.eval()

        # Append prefix
        texts = []
        for ix, sent in enumerate(sents):
            if inst_directions[ix] == constants.INST_BACKWARD:
                prefix = constants.ITN_PREFIX
            if inst_directions[ix] == constants.INST_FORWARD:
                prefix = constants.TN_PREFIX
            texts.append([prefix] + sent)

        # Apply the model
        prefix = constants.TN_PREFIX
        texts = [[prefix] + sent for sent in sents]
        encodings = self._tokenizer(
            texts, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt'
        )
        logits = self.model(**encodings.to(self.device)).logits
        pred_indexes = torch.argmax(logits, dim=-1).tolist()

        # Extract all_tag_preds
        all_tag_preds = []
        batch_size, max_len = encodings['input_ids'].size()
        for ix in range(batch_size):
            raw_tag_preds = [constants.ALL_TAG_LABELS[p] for p in pred_indexes[ix][1:]]
            tag_preds, previous_word_idx = [], None
            word_ids = encodings.word_ids(batch_index=ix)
            for jx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                elif word_idx != previous_word_idx:
                    tag_preds.append(raw_tag_preds[jx - 1])
                previous_word_idx = word_idx
            tag_preds = tag_preds[1:]
            all_tag_preds.append(tag_preds)

        # Postprocessing
        all_tag_preds = [
            self.postprocess_tag_preds(words, inst_dir, ps)
            for words, inst_dir, ps in zip(sents, inst_directions, all_tag_preds)
        ]

        # Decoding
        nb_spans, span_starts, span_ends = self.decode_tag_preds(all_tag_preds)

        return all_tag_preds, nb_spans, span_starts, span_ends

    def postprocess_tag_preds(self, words, inst_dir, preds):
        """ Function for postprocessing the raw tag predictions of the model. It
        corrects obvious mistakes in the tag predictions such as a TRANSFORM span
        starts with I_TRANSFORM_TAG (instead of B_TRANSFORM_TAG).

        Args:
            words: The words in the input text
            inst_dir: The direction of the instance (i.e., INST_BACKWARD or INST_FORWARD).
            preds: The raw tag predictions

        Returns: The processed raw tag predictions
        """
        final_preds = []
        for ix, p in enumerate(preds):
            # a TRANSFORM span starts with I_TRANSFORM_TAG
            if p == constants.I_PREFIX + constants.TRANSFORM_TAG:
                if ix == 0 or (not constants.TRANSFORM_TAG in final_preds[ix - 1]):
                    final_preds.append(constants.B_PREFIX + constants.TRANSFORM_TAG)
                    continue
            # a span has numbers but does not have TRANSFORM tags (for TN)
            if inst_dir == constants.INST_FORWARD:
                if has_numbers(words[ix]) and (not constants.TRANSFORM_TAG in p):
                    final_preds.append(constants.B_PREFIX + constants.TRANSFORM_TAG)
                    continue
            # Default
            final_preds.append(p)
        return final_preds

    def decode_tag_preds(self, tag_preds):
        """ Decoding the raw tag predictions to locate the semiotic spans in the
        input texts.

        Args:
            tag_preds: A list of list where each list contains the raw tag predictions for the corresponding input.

        Returns:
            nb_spans: A list of ints where each int indicates the number of semiotic spans in each input.
            span_starts: A list of lists where each list contains the starting locations of semiotic spans in an input.
            span_ends: A list of lists where each list contains the ending locations of semiotic spans in an input.
        """
        nb_spans, span_starts, span_ends = [], [], []
        for i, preds in enumerate(tag_preds):
            cur_nb_spans, cur_span_start = 0, None
            cur_span_starts, cur_span_ends = [], []
            for ix, pred in enumerate(preds + ['EOS']):
                if pred != constants.I_PREFIX + constants.TRANSFORM_TAG:
                    if not cur_span_start is None:
                        cur_nb_spans += 1
                        cur_span_starts.append(cur_span_start)
                        cur_span_ends.append(ix - 1)
                    cur_span_start = None
                if pred == constants.B_PREFIX + constants.TRANSFORM_TAG:
                    cur_span_start = ix
            nb_spans.append(cur_nb_spans)
            span_starts.append(cur_span_starts)
            span_ends.append(cur_span_ends)

        return nb_spans, span_starts, span_ends

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.data_path:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.data_path is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        start_time = perf_counter()
        logging.info(f'Creating {mode} dataset')
        input_file = cfg.data_path
        dataset = TextNormalizationTaggerDataset(
            input_file,
            self._tokenizer,
            cfg.mode,
            cfg.get('do_basic_tokenize', False),
            cfg.get('tagger_data_augmentation', False),
        )
        data_collator = DataCollatorForTokenClassification(self._tokenizer)
        dl = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, collate_fn=data_collator,
        )
        running_time = perf_counter() - start_time
        logging.info(f'Took {running_time} seconds')
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        return result
