# Copyright (c) 2019 NVIDIA Corporation
import torch

import nemo
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *
from .datasets import BertSentenceClassificationDataset,\
    BertJointIntentSlotDataset, \
    BertJointIntentSlotInferDataset


class BertSentenceClassificationDataLayer(DataLayerNM):
    """
    Creates the data layer to use for the task of sentence classification
    with pretrained model.

    All the data processing is done BertSentenceClassificationDataset.

    Args:
        input_file: file to sequence + label.
                    the first line is header (sentence [tab] label)
                    each line should be [sentence][tab][label]
        max_seq_length: max sequence length (minus 2 for [CLS] and [SEP])
        tokenizer: such as BERT tokenizer.
        num_samples: number of samples you want to use for the dataset.
                     if -1, use all dataset.
                     useful for testing.
    """

    @staticmethod
    def create_ports():
        output_ports = {
            "input_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_type_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels": NeuralType({
                0: AxisType(BatchTag),
            }),
        }
        return {}, output_ports

    def __init__(self,
                 path_to_data,
                 tokenizer,
                 max_seq_length,
                 num_samples=-1,
                 **kwargs):
        DataLayerNM.__init__(self, **kwargs)
        self._dataset = BertSentenceClassificationDataset(
            input_file=path_to_data,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            num_samples=num_samples)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


class BertJointIntentSlotDataLayer(DataLayerNM):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model.

    All the data processing is done in BertJointIntentSlotDataset.

    Args:
        input_file: file to sequence + label.
                    the first line is header (sentence [tab] label)
                    each line should be [sentence][tab][label]
        slot_file: file to slot labels, each line corresponding to
                   slot labels for a sentence in input_file. No header.
        max_seq_length: max sequence length (minus 2 for [CLS] and [SEP])
        tokenizer: such as BERT tokenizer.
        num_samples: number of samples you want to use for the dataset.
                     if -1, use all dataset.
                     useful for testing.
    """
    @staticmethod
    def create_ports():
        output_ports = {
            "input_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_type_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "token_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "intents": NeuralType({
                0: AxisType(BatchTag),
            }),
            "slots": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
        }
        return {}, output_ports

    def __init__(self,
                 path_to_data,
                 path_to_slot,
                 pad_label,
                 tokenizer,
                 max_seq_length,
                 num_samples=-1,
                 **kwargs):
        DataLayerNM.__init__(self, **kwargs)
        self._dataset = BertJointIntentSlotDataset(
            input_file=path_to_data,
            slot_file=path_to_slot,
            pad_label=pad_label,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            num_samples=num_samples)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


class BertJointIntentSlotInferDataLayer(DataLayerNM):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model. This is for

    All the data processing is done in BertJointIntentSlotDataset.

    Args:
        input_file: file to sequence + label.
                    the first line is header (sentence [tab] label)
                    each line should be [sentence][tab][label]
        slot_file: file to slot labels, each line corresponding to
                   slot labels for a sentence in input_file. No header.
        max_seq_length: max sequence length (minus 2 for [CLS] and [SEP])
        tokenizer: such as BERT tokenizer.
        num_samples: number of samples you want to use for the dataset.
                     if -1, use all dataset.
                     useful for testing.
    """
    @staticmethod
    def create_ports():
        output_ports = {
            "input_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_type_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "token_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        return {}, output_ports

    def __init__(self, queries, tokenizer, max_seq_length, **kwargs):
        DataLayerNM.__init__(self, **kwargs)
        self._dataset = BertJointIntentSlotInferDataset(
            queries=queries,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
