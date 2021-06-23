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

import os
import json
import torch
import numpy as np
import nltk
nltk.download('punkt')

from tqdm import tqdm
from nltk import word_tokenize
from collections import defaultdict
from typing import Dict, List, Optional
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf

from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoTokenizer

from nemo.core.classes.common import typecheck
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.core.neural_types import NeuralType
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.collections.nlp.models.neural_text_normalization.utils import *
from nemo.collections.nlp.models.neural_text_normalization.constants import *

__all__ = ['TextNormalizationTaggerModel']

class TextNormalizationTaggerModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        self.model = AutoModelForTokenClassification.from_pretrained(cfg.transformer)

    # Functions for inference
    @torch.no_grad()
    def _infer(
            self,
            sents: List[List[str]]
        ):
        """ Main function for Inference
        :param sents: A list of inputs tokenized by a basic tokenizer
                      (e.g., using nltk.word_tokenize()).
        """
        self.eval()

        # Apply the model
        prefix = TN_PREFIX
        texts = [[prefix] + sent for sent in sents]
        encodings = self._tokenizer(texts, is_split_into_words=True,
                                    padding=True, truncation=True,
                                    return_tensors='pt')
        logits = self.model(**encodings.to(self.device)).logits
        pred_indexes = torch.argmax(logits, dim=-1).tolist()

        # Extract all_tag_preds
        all_tag_preds = []
        batch_size, max_len = encodings['input_ids'].size()
        for ix in range(batch_size):
            raw_tag_preds = [ALL_TAG_LABELS[p] for p in pred_indexes[ix][1:]]
            tag_preds, previous_word_idx = [], None
            word_ids = encodings.word_ids(batch_index=ix)
            for jx, word_idx in enumerate(word_ids):
                if word_idx is None: continue
                elif word_idx != previous_word_idx:
                    tag_preds.append(raw_tag_preds[jx-1])
                previous_word_idx = word_idx
            tag_preds = tag_preds[1:]
            all_tag_preds.append(tag_preds)

        # Postprocessing
        all_tag_preds = [self.postprocess_tag_preds(words, ps)
                         for words, ps in zip(sents, all_tag_preds)]

        # Decoding
        nb_spans, span_starts, span_ends = self.decode_tag_preds(all_tag_preds)

        return nb_spans, span_starts, span_ends

    def postprocess_tag_preds(self, words, preds):
        final_preds = []
        for ix, p in enumerate(preds):
            # a TRANSFORM span starts with I_TRANSFORM_TAG
            if p == I_PREFIX + TRANSFORM_TAG:
                if ix == 0 or (not TRANSFORM_TAG in final_preds[ix-1]):
                    final_preds.append(B_PREFIX + TRANSFORM_TAG)
                    continue
            # a span has numbers but does not have TRANSFORM tags
            if has_numbers(words[ix]) and (not TRANSFORM_TAG in p):
                final_preds.append(B_PREFIX + TRANSFORM_TAG)
                continue
            final_preds.append(p)
        return final_preds

    def decode_tag_preds(self, tag_preds):
        nb_spans, span_starts, span_ends = [], [], []
        for i, preds in enumerate(tag_preds):
            cur_nb_spans, cur_span_start = 0, None
            cur_span_starts, cur_span_ends = [], []
            for ix, pred in enumerate(preds + ['EOS']):
                if pred != I_PREFIX + TRANSFORM_TAG:
                    if not cur_span_start is None:
                        cur_nb_spans += 1
                        cur_span_starts.append(cur_span_start)
                        cur_span_ends.append(ix-1)
                    cur_span_start = None
                if pred == B_PREFIX + TRANSFORM_TAG:
                    cur_span_start = ix
            nb_spans.append(cur_nb_spans)
            span_starts.append(cur_span_starts)
            span_ends.append(cur_span_ends)

        return nb_spans, span_starts, span_ends


    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or not train_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for test is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or not val_data_config.file:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for test is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.file is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode: str):
        pass
