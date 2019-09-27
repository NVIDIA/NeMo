# Copyright (c) 2019 NVIDIA Corporation
"""
If you want to add your own data layer, you should put its name in
__all__ so that it can be imported with 'from text_data_layers import *'
"""
__all__ = ['TextDataLayer',
           'BertSentenceClassificationDataLayer',
           'BertJointIntentSlotDataLayer',
           'BertJointIntentSlotInferDataLayer',
           'LanguageModelingDataLayer',
           'BertTokenClassificationDataLayer',
           'BertPretrainingDataLayer',
           'TranslationDataLayer']

import torch
from torch.utils import data as pt_data

import nemo
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *


class TextDataLayer(DataLayerNM):
    """
    Generic Text Data Layer NM which wraps PyTorch's dataset


    Args:
        dataset: a PyTorch dataset to wrap into Neural Module
        batch_size: 

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


class BertTokenClassificationDataLayer(TextDataLayer):
    @staticmethod
    def create_ports():
        input_ports = {}
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
                1: AxisType(TimeTag)
            }),
            "seq_ids": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def eval_preds(self, logits, seq_ids, tag_ids):
        return self._dataset.eval_preds(logits, seq_ids, tag_ids)


class BertPretrainingDataLayer(TextDataLayer):
    @staticmethod
    def create_ports():
        input_ports = {}
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
            "output_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "output_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

        return input_ports, output_ports


class TranslationDataLayer(TextDataLayer):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "src_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "src_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "tgt_ids": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "tgt_mask": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "labels": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            "sent_ids": NeuralType({
                0: AxisType(BatchTag)
            })
        }

        return input_ports, output_ports

    def __init__(self, dataset, **kwargs):
        TextDataLayer.__init__(self, None, **kwargs)
        
        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=lambda x: self._collate_fn(x),
            shuffle=sampler is None,
            sampler=sampler)

    def _collate_fn(self, x):
        src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids = x[0]
        src_ids = torch.Tensor(src_ids).long().to(self._device)
        src_mask = torch.Tensor(src_mask).float().to(self._device)
        tgt_ids = torch.Tensor(tgt_ids).long().to(self._device)
        tgt_mask = torch.Tensor(tgt_mask).float().to(self._device)
        labels = torch.Tensor(labels).long().to(self._device)
        sent_ids = torch.Tensor(sent_ids).long().to(self._device)
        return src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids

    @property
    def dataset(self):
        return None
    
    @property
    def data_iterator(self):
        return self._dataloader