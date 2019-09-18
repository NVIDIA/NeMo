# Copyright (c) 2019 NVIDIA Corporation
"""
If you want to add your own data layer, you should put its name in
__all__ so that it can be imported with 'from text_data_layers import *'
"""
__all__ = ['TextDataLayer',
           'BertSentenceClassificationDataLayer',
           'BertJointIntentSlotDataLayer',
           'BertJointIntentSlotInferDataLayer',
           'LanguageModelingDataLayer']

import torch

import nemo
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *


class TextDataLayer(DataLayerNM):
    """
    Generic Text Data Layer NM which wraps PyTorch's dataset


    Args:
        dataset: a PyTorch dataset to wrap into Neural Module

    """
    def __init__(self, dataset, **kwargs):
        DataLayerNM.__init__(self, **kwargs)
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None


class BertSentenceClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of sentence classification
    with pretrained model.

    All the data processing is done BertSentenceClassificationDataset.

    Args:
        dataset (BertSentenceClassificationDataset):
                the dataset that needs to be converted to DataLayerNM
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


class BertJointIntentSlotDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model.

    All the data processing is done in BertJointIntentSlotDataset.

    Args:
        dataset (BertJointIntentSlotDataset):
                the dataset that needs to be converted to DataLayerNM
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


class BertJointIntentSlotInferDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model. This is for

    All the data processing is done in BertJointIntentSlotInferDataset.

    Args:
        dataset (BertJointIntentSlotInferDataset):
                the dataset that needs to be converted to DataLayerNM
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


class LanguageModelingDataLayer(TextDataLayer):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "input_ids":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "input_mask":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels":
            NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }

        return input_ports, output_ports
