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

__all__ = ['TextNormalizationDecoderModel']

class TextNormalizationDecoderModel(NLPModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        self.model = AutoModelForTokenClassification.from_pretrained(cfg.transformer)

    # Functions for inference
    @torch.no_grad()
    def _infer(
            self,
            sents: List[List[str]],
            nb_spans: List[int],
            span_starts: List[List[int]],
            span_ends: List[List[int]],
        ):
        """ Main function for Inference
        :param sents: A list of inputs tokenized by a basic tokenizer
                      (e.g., using nltk.word_tokenize()).
        :param nb_spans: A list of ints where each int indicates the number of
                         semiotic spans in each input.
        :param span_starts: A list of lists where each list contains the
                            starting locations of semiotic spans in an input.
        :param span_ends: A list of lists where each list contains the ending
                          locations of semiotic spans in an input.
        """
        if sum(nb_spans) == 0: return [[]] * len(sents)
        model, tokenizer = self.model, self._tokenizer
        model_max_len = model.config.n_positions

        # Build all_inputs
        input_centers, all_inputs = [], []
        for ix, sent in enumerate(sents):
            cur_inputs = []
            for jx in range(nb_spans[ix]):
                cur_start = span_starts[ix][jx]
                cur_end = span_ends[ix][jx]
                ctx_left = sent[max(0, cur_start-DECODE_CTX_SIZE):cur_start]
                ctx_right = sent[cur_end+1:cur_end+1+DECODE_CTX_SIZE]
                span_words = sent[cur_start:cur_end+1]
                span_words_str = ' '.join(span_words)
                if is_url(span_words_str):
                    span_words_str = span_words_str.lower()
                input_centers.append(span_words_str)
                cur_inputs = [prefix] + ctx_left + ['<extra_id_0>'] + span_words_str.split(' ') + ['<extra_id_1>'] + ctx_right
                all_inputs.append(' '.join(cur_inputs))

        # Apply the decoding model
        batch = tokenizer(all_inputs, padding=True, return_tensors='pt')
        input_ids = batch['input_ids'].to(self.device)
        generated_ids = model.generate(input_ids, max_length=model_max_len)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Post processing
        generated_texts = self.postprocess_output_spans(input_centers, generated_texts)

        # Prepare final_texts
        final_texts, span_ctx = [], 0
        for nb_span in nb_spans:
            cur_texts = []
            for i in range(nb_span):
                cur_texts.append(generated_texts[span_ctx])
                span_ctx += 1
            final_texts.append(cur_texts)

        return final_texts

    def postprocess_output_spans(self, input_centers, output_spans):
        greek_spokens = list(GREEK_TO_SPOKEN.values())
        for ix, (_input, _output) in enumerate(zip(input_centers, output_spans)):
            # Handle URL
            if is_url(_input):
                output_spans[ix] = ' '.join(wordninja.split(_output))
                continue
            # Greek letters
            if _input in greek_spokens:
                output_spans[ix] = _input
        return output_spans

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
