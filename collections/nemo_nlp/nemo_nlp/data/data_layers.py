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

# from abc import abstractmethod
import sys

import torch
from torch.utils import data as pt_data

import nemo
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *

from .datasets import *


class TextDataLayer(DataLayerNM):
    """
    Generic Text Data Layer NM which wraps PyTorch's dataset

    Args:
        dataset_type: type of dataset used for this datalayer
        dataset_params (dict): all the params for the dataset
    """

    def __init__(self, dataset_type, dataset_params, **kwargs):
        super().__init__(**kwargs)
        if isinstance(dataset_type, str):
            dataset_type = getattr(sys.modules[__name__], dataset_type)
        self._dataset = dataset_type(**dataset_params)

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

    def __init__(self,
                 input_file,
                 tokenizer,
                 max_seq_length,
                 num_samples=-1,
                 shuffle=False,
                 batch_size=64,
                 dataset_type=BertSentenceClassificationDataset,
                 **kwargs):
        kwargs['batch_size'] = batch_size
        dataset_params = {'input_file': input_file,
                          'tokenizer': tokenizer,
                          'max_seq_length': max_seq_length,
                          'num_samples': num_samples,
                          'shuffle': shuffle}
        super().__init__(dataset_type, dataset_params, **kwargs)


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

    def __init__(self,
                 input_file,
                 slot_file,
                 pad_label,
                 tokenizer,
                 max_seq_length,
                 num_samples=-1,
                 shuffle=False,
                 batch_size=64,
                 dataset_type=BertJointIntentSlotDataset,
                 **kwargs):
        kwargs['batch_size'] = batch_size
        dataset_params = {'input_file': input_file,
                          'slot_file': slot_file,
                          'pad_label': pad_label,
                          'tokenizer': tokenizer,
                          'max_seq_length': max_seq_length,
                          'num_samples': num_samples,
                          'shuffle': shuffle}
        super().__init__(dataset_type, dataset_params, **kwargs)


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

    def __init__(self,
                 queries,
                 tokenizer,
                 max_seq_length,
                 batch_size=1,
                 dataset_type=BertJointIntentSlotInferDataset,
                 **kwargs):
        kwargs['batch_size'] = batch_size
        dataset_params = {'queries': queries,
                          'tokenizer': tokenizer,
                          'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, **kwargs)


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

    def __init__(self,
                 dataset,
                 tokenizer,
                 max_seq_length,
                 batch_step=128,
                 dataset_type=LanguageModelingDataset,
                 **kwargs):
        dataset_params = {'dataset': dataset,
                          'tokenizer': tokenizer,
                          'max_seq_length': max_seq_length,
                          'batch_step': batch_step}
        super().__init__(dataset_type, dataset_params, **kwargs)


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

    def __init__(self,
                 input_file,
                 tokenizer,
                 max_seq_length,
                 batch_size=64,
                 dataset_type=BertTokenClassificationDataset,
                 **kwargs):
        kwargs['batch_size'] = batch_size
        dataset_params = {'input_file': input_file,
                          'tokenizer': tokenizer,
                          'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, **kwargs)

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

    def __init__(self,
                 tokenizer,
                 dataset,
                 max_seq_length,
                 mask_probability,
                 batch_size=64,
                 **kwargs):
        kwargs['batch_size'] = batch_size
        dataset_params = {'tokenizer': tokenizer,
                          'dataset': dataset,
                          'max_seq_length': max_seq_length,
                          'mask_probability': mask_probability}
        super().__init__(BertPretrainingDataset, dataset_params, **kwargs)


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

    def __init__(self,
                 tokenizer_src,
                 tokenizer_tgt,
                 dataset_src,
                 dataset_tgt,
                 tokens_in_batch=1024,
                 clean=False,
                 dataset_type=TranslationDataset,
                 **kwargs):
        dataset_params = {'tokenizer_src': tokenizer_src,
                          'tokenizer_tgt': tokenizer_tgt,
                          'dataset_src': dataset_src,
                          'dataset_tgt': dataset_tgt,
                          'tokens_in_batch': tokens_in_batch,
                          'clean': clean}
        super().__init__(dataset_type, dataset_params, **kwargs)

        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(dataset=self._dataset,
                                              batch_size=1,
                                              collate_fn=self._collate_fn,
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
