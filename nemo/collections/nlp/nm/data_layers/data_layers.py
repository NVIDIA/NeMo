# Copyright (c) 2019 NVIDIA Corporation

# If you want to add your own data layer, you should put its name in
# __all__ so that it can be imported with 'from text_data_layers import *'


__all__ = [
    'GlueDataLayerClassificationDataLayer',
    'GlueDataLayerRegressionDataLayer',
    'BertJointIntentSlotDataLayer',
    'BertJointIntentSlotInferDataLayer',
    'BertPunctuationCapitalizationDataLayer',
    'BertPunctuationCapitalizationInferDataLayer',
    'BertPretrainingDataLayer',
    'BertPretrainingPreprocessedDataLayer',
    'BertSentenceClassificationDataLayer',
    'BertTokenClassificationDataLayer',
    'BertTokenClassificationInferDataLayer',
    'BertQuestionAnsweringDataLayer',
    'LanguageModelingDataLayer',
    'TextDataLayer',
    'TranslationDataLayer',
]

import os
import random
import sys

import h5py
import numpy as np
import torch
from torch.utils import data as pt_data

import nemo
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.collections.nlp.data import *
from nemo.core.neural_types import *


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

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels:
            0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self,
        input_file,
        tokenizer,
        max_seq_length,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        dataset_type=BertSentenceClassificationDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'input_file': input_file,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'num_samples': num_samples,
            'shuffle': shuffle,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertJointIntentSlotDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model.

    All the data processing is done in BertJointIntentSlotDataset.

    input_mask: used to ignore some of the input tokens like paddings

    loss_mask: used to mask and ignore tokens in the loss function

    subtokens_mask: used to ignore the outputs of unwanted tokens in
    the inference and evaluation like the start and end tokens

    Args:
        dataset (BertJointIntentSlotDataset):
            the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        intents:
            0: AxisType(BatchTag)

        slots:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "intents": NeuralType({0: AxisType(BatchTag)}),
            "slots": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        input_file,
        slot_file,
        pad_label,
        tokenizer,
        max_seq_length,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        dataset_type=BertJointIntentSlotDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'input_file': input_file,
            'slot_file': slot_file,
            'pad_label': pad_label,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'num_samples': num_samples,
            'shuffle': shuffle,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertJointIntentSlotInferDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model. This is for

    All the data processing is done in BertJointIntentSlotInferDataset.

    input_mask: used to ignore some of the input tokens like paddings

    loss_mask: used to mask and ignore tokens in the loss function

    subtokens_mask: used to ignore the outputs of unwanted tokens in
    the inference and evaluation like the start and end tokens

    Args:
        dataset (BertJointIntentSlotInferDataset):
            the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self, queries, tokenizer, max_seq_length, batch_size=1, dataset_type=BertJointIntentSlotInferDataset, **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {'queries': queries, 'tokenizer': tokenizer, 'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, **kwargs)


class LanguageModelingDataLayer(TextDataLayer):
    """
    Data layer for standard language modeling task.

    Args:
        dataset (str): path to text document with data
        tokenizer (TokenizerSpec): tokenizer
        max_seq_length (int): maximum allowed length of the text segments
        batch_step (int): how many tokens to skip between two successive
            segments of text when constructing batches
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of tokens which should be predicted from each of the
            corresponding tokens in input_ids; for left-to-right language
            modeling equals to input_ids shifted by 1 to the right
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self, dataset, tokenizer, max_seq_length, batch_step=128, dataset_type=LanguageModelingDataset, **kwargs
    ):
        dataset_params = {
            'dataset': dataset,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'batch_step': batch_step,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertTokenClassificationDataLayer(TextDataLayer):
    @property
    def output_ports(self):
        """Returns definitions of module output ports.

            input_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_type_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_mask:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            loss_mask:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            subtokens_mask:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            labels:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        text_file,
        label_file,
        tokenizer,
        max_seq_length,
        pad_label='O',
        label_ids=None,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        dataset_type=BertTokenClassificationDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'text_file': text_file,
            'label_file': label_file,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer,
            'num_samples': num_samples,
            'shuffle': shuffle,
            'pad_label': pad_label,
            'label_ids': label_ids,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertTokenClassificationInferDataLayer(TextDataLayer):
    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        queries,
        tokenizer,
        max_seq_length,
        batch_size=1,
        dataset_type=BertTokenClassificationInferDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {'queries': queries, 'tokenizer': tokenizer, 'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertPunctuationCapitalizationDataLayer(TextDataLayer):
    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        punct_labels:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        capit_labels:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "punct_labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "capit_labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        text_file,
        label_file,
        tokenizer,
        max_seq_length,
        pad_label='O',
        punct_label_ids=None,
        capit_label_ids=None,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        dataset_type=BertPunctuationCapitalizationDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'text_file': text_file,
            'label_file': label_file,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer,
            'num_samples': num_samples,
            'shuffle': shuffle,
            'pad_label': pad_label,
            'punct_label_ids': punct_label_ids,
            'capit_label_ids': capit_label_ids,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertPunctuationCapitalizationInferDataLayer(TextDataLayer):
    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        queries,
        tokenizer,
        max_seq_length,
        batch_size=1,
        dataset_type=BertTokenClassificationInferDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {'queries': queries, 'tokenizer': tokenizer, 'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertQuestionAnsweringDataLayer(TextDataLayer):
    """
    Creates the data layer to use for Question Answering classification task.

    Args:
        data_dir (str): Directory that contains train.*.json and dev.*.json.
        tokenizer (obj): Tokenizer object, e.g. NemoBertTokenizer.
        version_2_with_negative (bool): True if training should allow
            unanswerable questions.
        doc_stride (int): When splitting up a long document into chunks,
            how much stride to take between chunks.
        max_query_length (iny): All training files which have a duration less
            than min_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        max_seq_length (int): All training files which have a duration more
            than max_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        mode (str): Use "train" or "dev" to define between
            training and evaluation.
        batch_size (int): Batch size. Defaults to 64.
        dataset_type (class): Question Answering class.
            Defaults to SquadDataset.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

            input_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_type_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_mask:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            start_positions:
                0: AxisType(BatchTag)

            end_positions:
                0: AxisType(BatchTag)

            unique_ids:
                0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "start_positions": NeuralType({0: AxisType(BatchTag)}),
            "end_positions": NeuralType({0: AxisType(BatchTag)}),
            "unique_ids": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        version_2_with_negative,
        doc_stride,
        max_query_length,
        max_seq_length,
        mode="train",
        batch_size=64,
        dataset_type=SquadDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'data_dir': data_dir,
            'mode': mode,
            'tokenizer': tokenizer,
            'version_2_with_negative': version_2_with_negative,
            'max_query_length': max_query_length,
            'max_seq_length': max_seq_length,
            'doc_stride': doc_stride,
        }

        super().__init__(dataset_type, dataset_params, **kwargs)


class BertPretrainingDataLayer(TextDataLayer):
    """
    Data layer for masked language modeling task.

    Args:
        tokenizer (TokenizerSpec): tokenizer
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        mask_probability (float): probability of masking input sequence tokens
        batch_size (int): batch size in segments
        short_seeq_prob (float): Probability of creating sequences which are
            shorter than the maximum length.
            Defualts to 0.1.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids: indices of token types (e.g., sentences A & B in BERT)
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_ids: indices of output tokens which should be predicted
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_mask: bool tensor with 0s in place of tokens to be excluded
            from loss calculation
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of classes to be predicted from [CLS] token of text
            segments (e.g, 0 or 1 in next sentence prediction task)
            0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self, tokenizer, dataset, max_seq_length, mask_probability, short_seq_prob=0.1, batch_size=64, **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'tokenizer': tokenizer,
            'dataset': dataset,
            'max_seq_length': max_seq_length,
            'mask_probability': mask_probability,
            'short_seq_prob': short_seq_prob,
        }
        super().__init__(BertPretrainingDataset, dataset_params, **kwargs)


class BertPretrainingPreprocessedDataLayer(DataLayerNM):
    """
    Data layer for masked language modeling task.

    Args:
        tokenizer (TokenizerSpec): tokenizer
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        mask_probability (float): probability of masking input sequence tokens
        batch_size (int): batch size in segments
        short_seeq_prob (float): Probability of creating sequences which are
            shorter than the maximum length.
            Defualts to 0.1.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids: indices of token types (e.g., sentences A & B in BERT)
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_ids: indices of output tokens which should be predicted
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_mask: bool tensor with 0s in place of tokens to be excluded
            from loss calculation
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of classes to be predicted from [CLS] token of text
            segments (e.g, 0 or 1 in next sentence prediction task)
            0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(self, dataset, max_pred_length, batch_size=64, training=True, **kwargs):

        if os.path.isdir(dataset):
            self.files = [
                os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))
            ]
        else:
            self.files = [dataset]
        self.files.sort()
        self.num_files = len(self.files)
        self.batch_size = batch_size
        self.max_pred_length = max_pred_length
        self.training = training
        total_length = 0
        for f in self.files:
            fp = h5py.File(f, 'r')
            total_length += len(fp['input_ids'])
            fp.close()
        self.total_length = total_length
        super().__init__(**kwargs)

    def _collate_fn(self, x):
        num_components = len(x[0])
        components = [[] for _ in range(num_components)]
        batch_size = len(x)
        for i in range(batch_size):
            for j in range(num_components):
                components[j].append(x[i][j])
        src_ids, src_segment_ids, src_mask, tgt_ids, tgt_mask, sent_ids = [np.stack(x, axis=0) for x in components]
        src_ids = torch.Tensor(src_ids).long().to(self._device)
        src_segment_ids = torch.Tensor(src_segment_ids).long().to(self._device)
        src_mask = torch.Tensor(src_mask).long().to(self._device)
        tgt_ids = torch.Tensor(tgt_ids).long().to(self._device)
        tgt_mask = torch.Tensor(tgt_mask).long().to(self._device)
        sent_ids = torch.Tensor(sent_ids).long().to(self._device)
        return src_ids, src_segment_ids, src_mask, tgt_ids, tgt_mask, sent_ids

    def __len__(self):
        return self.total_length

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        while True:
            if self.training:
                random.shuffle(self.files)
            for f_id in range(self.num_files):
                data_file = self.files[f_id]
                train_data = BertPretrainingPreprocessedDataset(
                    input_file=data_file, max_pred_length=self.max_pred_length
                )
                train_sampler = pt_data.RandomSampler(train_data)
                train_dataloader = pt_data.DataLoader(
                    dataset=train_data,
                    batch_size=self.batch_size,
                    collate_fn=self._collate_fn,
                    shuffle=train_sampler is None,
                    sampler=train_sampler,
                )
                for x in train_dataloader:
                    yield x


class TranslationDataLayer(TextDataLayer):
    """
    Data layer for neural machine translation from source (src) language to
    target (tgt) language.

    Args:
        tokenizer_src (TokenizerSpec): source language tokenizer
        tokenizer_tgt (TokenizerSpec): target language tokenizer
        dataset_src (str): path to source data
        dataset_tgt (str): path to target data
        tokens_in_batch (int): maximum allowed number of tokens in batches,
            batches will be constructed to minimize the use of <pad> tokens
        clean (bool): whether to use parallel data cleaning such as removing
            pairs with big difference in sentences length, removing pairs with
            the same tokens in src and tgt, etc; useful for training data layer
            and should not be used in evaluation data layer
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        src_ids: indices of tokens which correspond to source sentences
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        src_mask: bool tensor with 0s in place of source tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        tgt_ids: indices of tokens which correspond to target sentences
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        tgt_mask: bool tensor with 0s in place of target tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of tokens which should be predicted from each of the
            corresponding target tokens in tgt_ids; for standard neural
            machine translation equals to tgt_ids shifted by 1 to the right
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        sent_ids: indices of the sentences in a batch; important for
            evaluation with external metrics, such as SacreBLEU
            0: AxisType(BatchTag)

        """
        return {
            "src_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "src_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "tgt_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "tgt_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "sent_ids": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self,
        tokenizer_src,
        tokenizer_tgt,
        dataset_src,
        dataset_tgt,
        tokens_in_batch=1024,
        clean=False,
        dataset_type=TranslationDataset,
        **kwargs
    ):
        dataset_params = {
            'tokenizer_src': tokenizer_src,
            'tokenizer_tgt': tokenizer_tgt,
            'dataset_src': dataset_src,
            'dataset_tgt': dataset_tgt,
            'tokens_in_batch': tokens_in_batch,
            'clean': clean,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)

        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=self._dataset, batch_size=1, collate_fn=self._collate_fn, shuffle=sampler is None, sampler=sampler
        )

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


class GlueDataLayerClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the GLUE classification tasks,
    more details here: https://gluebenchmark.com/tasks

    All the data processing is done in GLUEDataset.

    Args:
        dataset_type (GLUEDataset):
                the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

            input_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_type_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_mask:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            labels:
                0: AxisType(CategoricalTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(CategoricalTag)}),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_seq_length,
        processor,
        evaluate=False,
        token_params={},
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        dataset_type=GLUEDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'data_dir': data_dir,
            'output_mode': 'classification',
            'processor': processor,
            'evaluate': evaluate,
            'token_params': token_params,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
        }

        super().__init__(dataset_type, dataset_params, **kwargs)


class GlueDataLayerRegressionDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the GLUE STS-B regression task,
    more details here: https://gluebenchmark.com/tasks

    All the data processing is done in GLUEDataset.

    Args:
        dataset_type (GLUEDataset):
                the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

            input_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_type_ids:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            input_mask:
                0: AxisType(BatchTag)

                1: AxisType(TimeTag)

            labels:
                0: AxisType(RegressionTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(RegressionTag)}),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_seq_length,
        processor,
        evaluate=False,
        token_params={},
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        dataset_type=GLUEDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'data_dir': data_dir,
            'output_mode': 'regression',
            'processor': processor,
            'evaluate': evaluate,
            'token_params': token_params,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
        }

        super().__init__(dataset_type, dataset_params, **kwargs)
