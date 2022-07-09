# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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
import pickle

import numpy as np
import torch
from tqdm import trange

from nemo.collections.nlp.data.question_answering.data_processor.qa_processing import (
    EVALUATION_MODE,
    INFERENCE_MODE,
    TRAINING_MODE,
)
from nemo.collections.nlp.data.question_answering.dataset.qa_dataset import QADataset
from nemo.collections.nlp.data.question_answering.input_features.qa_input_features import S2SQAInputFeatures
from nemo.utils import logging


class S2SQADataset(QADataset):
    """ Creates a Dataset for T5/BART architecture based Generative QA """

    def __init__(
        self,
        data_file: str,
        processor: object,
        tokenizer: object,
        keep_doc_spans: str = False,
        doc_stride: int = 128,
        max_query_length: int = 64,
        max_seq_length: int = 512,
        max_answer_length: int = 64,
        num_samples: int = -1,
        mode: str = TRAINING_MODE,
        use_cache: bool = False,
    ):
        super().__init__(data_file=data_file, processor=processor, tokenizer=tokenizer)

        self.keep_doc_spans = keep_doc_spans
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.num_samples = num_samples
        self.mode = mode
        self.use_cache = use_cache

        self._check_valid_mode()

        # get examples from processor and keep according to limit
        self.examples = self.processor.get_examples()
        if num_samples == 0:
            raise ValueError(
                f"num_samples has to be positive or -1 (to use the entire dataset), however got {num_samples}."
            )
        elif num_samples > 0:
            self.examples = self.examples[:num_samples]

        # create cached filename to load/save features as per flag
        vocab_size = getattr(self.tokenizer, "vocab_size", 0)
        self.cached_features_file = (
            self.data_file
            + '_cache'
            + '_{}_{}_{}_{}_{}_{}_{}'.format(
                self.mode,
                self.tokenizer.name,
                str(vocab_size),
                str(self.max_query_length),
                str(self.max_seq_length),
                str(self.max_answer_length),
                str(self.num_samples),
            )
        )

        if use_cache and os.path.exists(self.cached_features_file):
            self._load_features_from_cache()
        else:
            self._convert_examples_to_features()
            if use_cache:
                self._dump_features_to_cache()

        logging.info("Converting dict features into object features")
        for i in trange(len(self.features)):
            self.features[i] = S2SQAInputFeatures(**self.features[i])

    def _convert_examples_to_features(self):
        """ Converts loaded examples to features """

        logging.info(f"Preprocessing data into features.")

        has_groundtruth = self.mode != INFERENCE_MODE
        unique_id = 1000000000
        self.features = []
        context_prefix = "context: "
        context_prefix_tokens = self.tokenizer.tokenizer.tokenize(context_prefix)

        for example_index in trange(len(self.examples)):
            if example_index % 1000 == 0:
                S2SQADataset.check_if_sufficient_memory()

            example = self.examples[example_index]

            query_tokens, formatted_query = self._prep_query(example)
            context_tokens, context_spans = self._prep_context(example, query_tokens, context_prefix_tokens)

            unique_id = self._encode_all_context_spans(
                unique_id, context_spans, context_tokens, formatted_query, example, example_index, has_groundtruth,
            )

        # delete self.examples during training mode to save memory
        if self.mode == TRAINING_MODE:
            self.examples = []
            del self.processor

    def _prep_query(self, example):
        formatted_query = f" question: {example.question_text}"
        query_tokens = self.tokenizer.tokenizer.tokenize(formatted_query)[: self.max_query_length]

        return query_tokens, formatted_query

    def _prep_context(self, example, query_tokens, context_prefix_tokens):
        # -1 accounts for </s> token in T5/BART
        context_tokens = self.tokenizer.tokenizer.tokenize(example.context_text)
        max_context_length = self.max_seq_length - len(query_tokens) - len(context_prefix_tokens) - 1
        context_spans = S2SQADataset.get_docspans(context_tokens, max_context_length, self.doc_stride)
        context_spans = tuple(context_spans)

        return context_tokens, context_spans

    def _encode_all_context_spans(
        self, unique_id, context_spans, context_tokens, formatted_query, example, example_index, has_groundtruth,
    ):
        for context_span_idx, context_span in enumerate(context_spans):

            # format query and context span text
            context_span_tokens = context_tokens[context_span.start : context_span.start + context_span.length]
            context_span_text = self.tokenizer.tokenizer.convert_tokens_to_string(context_span_tokens)
            source = f"context: {context_span_text}{formatted_query}"

            # encode input
            encoded_input_dict = self.tokenizer.tokenizer(
                source, truncation=True, max_length=self.max_seq_length, padding="max_length", return_tensors="pt",
            )
            input_ids = torch.squeeze(encoded_input_dict["input_ids"])
            input_attn_mask = torch.squeeze(encoded_input_dict["attention_mask"])

            # encode output based on if answer present in context
            labels = self._encode_answer(has_groundtruth, example, context_span_text)

            # create dictionary features
            feature = {
                "unique_id": unique_id,
                "input_ids": input_ids,
                "input_attn_mask": input_attn_mask,
                "labels": labels,
                "example_index": example_index,
                "context_span_index": context_span_idx,
                "is_impossible": example.is_impossible,
            }

            self.features.append(feature)
            unique_id += 1

        return unique_id

    def _encode_answer(self, has_groundtruth, example, context_span_text):
        if has_groundtruth:
            if not example.is_impossible:
                target = example.answer_text if example.answer_text in context_span_text else ""
            else:
                target = ""
        else:
            target = None

        if target is not None:
            encoded_output_dict = self.tokenizer.tokenizer(
                target, truncation=True, max_length=self.max_answer_length, padding="max_length", return_tensors="pt",
            )
            labels = torch.squeeze(encoded_output_dict["input_ids"])
            labels[labels == self.tokenizer.tokenizer.pad_token_id] = -100
        else:
            labels = None

        return labels

    def _load_features_from_cache(self):
        """ Loads pickled features from the file """

        logging.info(f"loading from {self.cached_features_file}")

        # delete self.examples during training mode to save memory
        if self.mode == TRAINING_MODE:
            del self.examples
            del self.processor

        with open(self.cached_features_file, "rb") as reader:
            self.features = pickle.load(reader)

    def _dump_features_to_cache(self):
        """ Pickles features to file """

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if master_device:
            logging.info(f"Saving train features into cached file {self.cached_features_file}")
            with open(self.cached_features_file, "wb") as writer:
                pickle.dump(self.features, writer)

    def _check_valid_mode(self):
        """ Checks if provided mode is in given three options """

        if self.mode not in [TRAINING_MODE, EVALUATION_MODE, INFERENCE_MODE]:
            raise ValueError(
                f"mode should be either {TRAINING_MODE}, {EVALUATION_MODE}, {INFERENCE_MODE} but got {self.mode}"
            )
        else:
            return

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        feature = self.features[idx]
        if self.mode == INFERENCE_MODE:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
            )
        else:
            return (
                np.array(feature.input_ids),
                np.array(feature.input_attn_mask),
                np.array(feature.unique_id),
                np.array(feature.labels),
            )
