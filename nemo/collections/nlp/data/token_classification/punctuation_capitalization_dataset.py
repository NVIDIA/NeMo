# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

__all__ = ['BertPunctuationCapitalizationDataset', 'BertPunctuationCapitalizationInferDataset']

import itertools
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_label_stats, get_stats
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, Index, LabelsType, MaskType, NeuralType
from nemo.core.neural_types.elements import BoolType
from nemo.utils import logging


def get_features(
    queries: List[str],
    max_seq_length: int,
    tokenizer: TokenizerSpec,
    punct_label_ids: dict = None,
    capit_label_ids: dict = None,
    pad_label: str = 'O',
    punct_labels_lines=None,
    capit_labels_lines=None,
    ignore_extra_tokens=False,
    ignore_start_end: Optional[bool] = False,
):
    """
    Processes the data and returns features.

    Args:
        queries: text sequences
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as AutoTokenizer
        pad_label: pad value use for labels. By default, it's the neutral label.
        punct_label_ids: dict to map punctuation labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        capit_label_ids: dict to map labels to label ids. Starts
            with pad_label->0 and then increases in alphabetical order.
            Required for training and evaluation, not needed for inference.
        punct_labels: list of labels for every word in a sequence (str)
        capit_labels: list of labels for every word in a sequence (str)
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask

    Returns:
        all_input_ids: input ids for all tokens
        all_segment_ids: token type ids
        all_input_mask: attention mask to use for BERT model
        all_subtokens_mask: masks out all subwords besides the first one
        all_loss_mask: loss mask to mask out tokens during training
        punct_all_labels: all labels for punctuation task (ints)
        capit_all_labels: all labels for capitalization task (ints)
        punct_label_ids: label (str) to id (int) map for punctuation task
        capit_label_ids: label (str) to id (int) map for capitalization task
    """
    all_subtokens = []
    all_loss_mask = []
    all_subtokens_mask = []
    all_segment_ids = []
    all_input_ids = []
    all_input_mask = []
    sent_lengths = []
    punct_all_labels = []
    capit_all_labels = []
    with_label = False

    if punct_labels_lines and capit_labels_lines:
        with_label = True

    for i, query in enumerate(queries):
        words = query.strip().split()

        # add bos token
        subtokens = [tokenizer.cls_token]
        loss_mask = [1 - ignore_start_end]
        subtokens_mask = [0]
        if with_label:
            pad_id = punct_label_ids[pad_label]
            punct_labels = [pad_id]
            punct_query_labels = [punct_label_ids[lab] for lab in punct_labels_lines[i]]

            capit_labels = [pad_id]
            capit_query_labels = [capit_label_ids[lab] for lab in capit_labels_lines[i]]

        for j, word in enumerate(words):
            word_tokens = tokenizer.text_to_tokens(word)
            subtokens.extend(word_tokens)

            loss_mask.append(1)
            loss_mask.extend([int(not ignore_extra_tokens)] * (len(word_tokens) - 1))

            subtokens_mask.append(1)
            subtokens_mask.extend([0] * (len(word_tokens) - 1))

            if with_label:
                punct_labels.extend([punct_query_labels[j]] * len(word_tokens))
                capit_labels.extend([capit_query_labels[j]] * len(word_tokens))

        # add eos token
        subtokens.append(tokenizer.sep_token)
        loss_mask.append(1 - ignore_start_end)
        subtokens_mask.append(0)
        sent_lengths.append(len(subtokens))
        all_subtokens.append(subtokens)
        all_loss_mask.append(loss_mask)
        all_subtokens_mask.append(subtokens_mask)
        all_input_mask.append([1] * len(subtokens))

        if with_label:
            punct_labels.append(pad_id)
            punct_all_labels.append(punct_labels)
            capit_labels.append(pad_id)
            capit_all_labels.append(capit_labels)

    max_seq_length = min(max_seq_length, max(sent_lengths))
    logging.info(f'Max length: {max_seq_length}')
    get_stats(sent_lengths)
    too_long_count = 0

    for i, subtokens in enumerate(all_subtokens):
        if len(subtokens) > max_seq_length:
            subtokens = [tokenizer.cls_token] + subtokens[-max_seq_length + 1 :]
            all_input_mask[i] = [1] + all_input_mask[i][-max_seq_length + 1 :]
            all_loss_mask[i] = [int(not ignore_start_end)] + all_loss_mask[i][-max_seq_length + 1 :]
            all_subtokens_mask[i] = [0] + all_subtokens_mask[i][-max_seq_length + 1 :]

            if with_label:
                punct_all_labels[i] = [pad_id] + punct_all_labels[i][-max_seq_length + 1 :]
                capit_all_labels[i] = [pad_id] + capit_all_labels[i][-max_seq_length + 1 :]
            too_long_count += 1

        all_input_ids.append(tokenizer.tokens_to_ids(subtokens))

        if len(subtokens) < max_seq_length:
            extra = max_seq_length - len(subtokens)
            all_input_ids[i] = all_input_ids[i] + [0] * extra
            all_loss_mask[i] = all_loss_mask[i] + [0] * extra
            all_subtokens_mask[i] = all_subtokens_mask[i] + [0] * extra
            all_input_mask[i] = all_input_mask[i] + [0] * extra

            if with_label:
                punct_all_labels[i] = punct_all_labels[i] + [pad_id] * extra
                capit_all_labels[i] = capit_all_labels[i] + [pad_id] * extra

        all_segment_ids.append([0] * max_seq_length)

    logging.info(f'{too_long_count} are longer than {max_seq_length}')

    for i in range(min(len(all_input_ids), 5)):
        logging.info("*** Example ***")
        logging.info("i: %s" % (i))
        logging.info("subtokens: %s" % " ".join(list(map(str, all_subtokens[i]))))
        logging.info("loss_mask: %s" % " ".join(list(map(str, all_loss_mask[i]))))
        logging.info("input_mask: %s" % " ".join(list(map(str, all_input_mask[i]))))
        logging.info("subtokens_mask: %s" % " ".join(list(map(str, all_subtokens_mask[i]))))
        if with_label:
            logging.info("punct_labels: %s" % " ".join(list(map(str, punct_all_labels[i]))))
            logging.info("capit_labels: %s" % " ".join(list(map(str, capit_all_labels[i]))))

    return (
        all_input_ids,
        all_segment_ids,
        all_input_mask,
        all_subtokens_mask,
        all_loss_mask,
        punct_all_labels,
        capit_all_labels,
        punct_label_ids,
        capit_label_ids,
    )


class BertPunctuationCapitalizationDataset(Dataset):
    """
    Creates dataset to use during training for punctuaion and capitalization tasks with a pretrained model.
    For dataset to use during inference without labels, see BertPunctuationCapitalizationInferDataset.

    Args:
        text_file: file to sequences, each line should a sentence, no header.
        label_file: file to labels, each line corresponds to word labels for a sentence in the text_file. No header.
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        tokenizer: such as AutoTokenizer
        num_samples: number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        pad_label: pad value use for labels.
            by default, it's the neutral label.
        punct_label_ids and capit_label_ids (dict):
            dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order
            For dev set use label_ids generated during training to support
            cases when not all labels are present in the dev set.
            For training set label_ids should be None or loaded from cache
        ignore_extra_tokens: whether to ignore extra tokens in the loss_mask
        ignore_start_end: whether to ignore bos and eos tokens in the loss_mask
        use_cache: whether to use processed data cache or not
        get_label_frequencies: whether to generate label frequencies
        punct_label_ids_file and capit_label_ids_file: name of the files to save in .nemo
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports. """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'loss_mask': NeuralType(('B', 'T'), MaskType()),
            'punct_labels': NeuralType(('B', 'T'), LabelsType()),
            'capit_labels': NeuralType(('B', 'T'), LabelsType()),
        }

    def __init__(
        self,
        text_file: str,
        label_file: str,
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        num_samples: int = -1,
        pad_label: str = 'O',
        punct_label_ids: Dict[str, int] = None,
        capit_label_ids: Dict[str, int] = None,
        ignore_extra_tokens: bool = False,
        ignore_start_end: bool = False,
        use_cache: bool = True,
        get_label_frequencies: bool = False,
        punct_label_ids_file: str = 'punct_label_ids.csv',
        capit_label_ids_file: str = 'capit_label_ids.csv',
    ):
        """ Initializes BertPunctuationCapitalizationDataset. """

        if not (os.path.exists(text_file) and os.path.exists(label_file)):
            raise FileNotFoundError(
                f'{text_file} or {label_file} not found. The data should be splitted into 2 files: text.txt and \
                labels.txt. Each line of the text.txt file contains text sequences, where words are separated with \
                spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are \
                separated with spaces. Each line of the files should follow the format:  \
                   [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
                   [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )

        # Cache features
        data_dir = os.path.dirname(text_file)
        filename = os.path.basename(text_file)

        if not filename.endswith('.txt'):
            raise ValueError("{text_file} should have extension .txt")

        filename = filename[:-4]
        vocab_size = getattr(tokenizer, "vocab_size", 0)
        features_pkl = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                filename, tokenizer.name, str(max_seq_length), str(vocab_size), str(num_samples)
            ),
        )

        self.punct_label_ids_file = os.path.join(data_dir, punct_label_ids_file)
        self.capit_label_ids_file = os.path.join(data_dir, capit_label_ids_file)

        master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        cache_files_exist = (
            os.path.exists(features_pkl)
            and os.path.exists(self.punct_label_ids_file)
            and os.path.exists(self.capit_label_ids_file)
        )
        features = None
        if master_device and not (cache_files_exist and use_cache):
            if num_samples == 0:
                raise ValueError("num_samples has to be positive", num_samples)
            logging.info(f'Processing {text_file}')
            with open(text_file, 'r') as f:
                text_lines = f.readlines()

            # Collect all possible labels
            punct_unique_labels = set()
            capit_unique_labels = set()
            punct_labels_lines = []
            capit_labels_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip().split()

                    # extract punctuation and capitalization labels
                    punct_line, capit_line = zip(*line)
                    punct_labels_lines.append(punct_line)
                    capit_labels_lines.append(capit_line)

                    punct_unique_labels.update(punct_line)
                    capit_unique_labels.update(capit_line)

            if len(punct_labels_lines) != len(text_lines):
                raise ValueError("Labels file should contain labels for every word")

            dataset = list(zip(text_lines, punct_labels_lines, capit_labels_lines))

            if num_samples > 0:
                dataset = dataset[:num_samples]

            dataset = list(zip(*dataset))
            text_lines = dataset[0]
            punct_labels_lines = dataset[1]
            capit_labels_lines = dataset[2]

            # for dev/test sets use label mapping from training set
            if punct_label_ids:
                if len(punct_label_ids) != len(punct_unique_labels):
                    logging.info(
                        'Not all labels from the specified'
                        + 'label_ids dictionary are present in the'
                        + 'current dataset. Using the provided'
                        + 'label_ids dictionary.'
                    )
                else:
                    logging.info('Using the provided label_ids dictionary.')
            else:
                logging.info(
                    'Creating a new label to label_id dictionary.'
                    + ' It\'s recommended to use label_ids generated'
                    + ' during training for dev/test sets to avoid'
                    + ' errors if some labels are not'
                    + ' present in the dev/test sets.'
                    + ' For training set label_ids should be None.'
                )

                def create_label_ids(unique_labels, pad_label=pad_label):
                    label_ids = {pad_label: 0}
                    if pad_label in unique_labels:
                        unique_labels.remove(pad_label)
                    for label in sorted(unique_labels):
                        label_ids[label] = len(label_ids)
                    return label_ids

                punct_label_ids = create_label_ids(punct_unique_labels)
                capit_label_ids = create_label_ids(capit_unique_labels)

            self._save_label_ids(punct_label_ids, self.punct_label_ids_file)
            self._save_label_ids(capit_label_ids, self.capit_label_ids_file)

            features = get_features(
                text_lines,
                max_seq_length,
                tokenizer,
                pad_label=pad_label,
                punct_labels_lines=punct_labels_lines,
                capit_labels_lines=capit_labels_lines,
                punct_label_ids=punct_label_ids,
                capit_label_ids=capit_label_ids,
                ignore_extra_tokens=ignore_extra_tokens,
                ignore_start_end=ignore_start_end,
            )

            pickle.dump(features, open(features_pkl, "wb"))
            logging.info(f'Features saved to {features_pkl}')

        # wait until the master process writes to the processed data files
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if features is None:
            features = pickle.load(open(features_pkl, 'rb'))
            logging.info(f'Features restored from {features_pkl}')

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_subtokens_mask = features[3]
        self.all_loss_mask = features[4]
        self.punct_all_labels = features[5]
        self.capit_all_labels = features[6]
        self.punct_label_ids = features[7]
        self.capit_label_ids = features[8]

        if get_label_frequencies:
            self.punct_label_frequencies = self._calculate_label_frequencies(self.punct_all_labels, data_dir, 'punct')
            self.capit_label_frequencies = self._calculate_label_frequencies(self.capit_all_labels, data_dir, 'capit')

    def _calculate_label_frequencies(self, all_labels: List[int], data_dir: str, name: str) -> Dict[str, float]:
        """ Calculates labels frequencies """
        merged_labels = itertools.chain.from_iterable(all_labels)
        logging.info('Three most popular labels')
        _, label_frequencies, _ = get_label_stats(merged_labels, data_dir + '/label_count_' + name + '.tsv')
        return label_frequencies

    def _save_label_ids(self, label_ids: Dict[str, int], filename: str) -> None:
        """ Saves label ids map to a file """
        with open(filename, 'w') as out:
            labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
            out.write('\n'.join(labels))
            logging.info(f'Labels: {label_ids}')
            logging.info(f'Labels mapping saved to : {out.name}')

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_subtokens_mask[idx]),
            np.array(self.all_loss_mask[idx]),
            np.array(self.punct_all_labels[idx]),
            np.array(self.capit_all_labels[idx]),
        )


def _get_subtokens_and_subtokens_mask(query: str, tokenizer: TokenizerSpec) -> Tuple[List[str], List[int]]:
    """
    Tokenizes input query into subtokens and creates subtokens mask. Subtokens mask is an array of the same length as
    subtokens array and contains zeros and ones in which. If element of mask equals 1, then corresponding subtoken in
    subtokens array is first subtoken in some word
    Args:
        query: a string that will be tokenized
        tokenizer: an instance of tokenizer
    Returns:
        subtokens: list of subtokens
        subtokens_mask: list of ints
    """
    words = query.strip().split()
    subtokens = []
    subtokens_mask = []
    for j, word in enumerate(words):
        word_tokens = tokenizer.text_to_tokens(word)
        subtokens.extend(word_tokens)
        subtokens_mask.append(1)
        subtokens_mask.extend([0] * (len(word_tokens) - 1))
    return subtokens, subtokens_mask


def _check_max_seq_length_and_margin_and_step(max_seq_length: int, margin: int, step: int):
    """
    Checks values of ``max_seq_length``, ``margin``, and ``step``.
    Args:
        max_seq_length: a segment length with ``[CLS]`` and ``[SEP]`` tokens
        margin: a number of input tokens near edges of segments which are not used in punctuation and capitalization
            prediction.
        step: offset of consequent segments.
    Returns:
        None
    """
    if max_seq_length < 3:
        raise ValueError(
            f"Parameter `max_seq_length={max_seq_length}` cannot be less than 3 because `max_seq_length` is a length "
            f"of a segment with [CLS] and [SEP] tokens."
        )
    if margin >= (max_seq_length - 2) // 2 and margin > 0 or margin < 0:
        raise ValueError(
            f"Parameter `margin` has to be not negative and less than `(max_seq_length - 2) // 2`. Don't forget about "
            f"CLS and EOS tokens in the beginning and the end of segment. margin={margin}, "
            f"max_seq_length={max_seq_length}"
        )
    if step <= 0:
        raise ValueError(f"Parameter `step` has to be positive whereas step={step}")
    if step > max_seq_length - 2 - 2 * margin:
        logging.warning(
            f"Parameter step={step} is too big. It will be reduced to `min(max_seq_length, <maximum query length> + 2) "
            f"- 2 - 2 * margin`."
        )


def get_features_infer(
    queries: List[str],
    tokenizer: TokenizerSpec,
    max_seq_length: int = 64,
    step: Optional[int] = 8,
    margin: Optional[int] = 16,
) -> Tuple[
    List[List[int]], List[List[int]], List[List[int]], List[List[int]], List[int], List[int], List[bool], List[bool],
]:
    """
    Processes the data and returns features.

    Args:
        queries: text sequences
        tokenizer: such as AutoTokenizer
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        step: relative shift of consequent segments into which long queries are split. Long queries are split into
            segments which can overlap. Parameter ``step`` controls such overlapping. Imagine that queries are
            tokenized into characters, ``max_seq_length=5``, and ``step=2``. In such a case query "hello" is
            tokenized into segments ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]``.
        margin: number of subtokens near edges of segments which are not used for punctuation and capitalization
            prediction. The first segment does not have left margin and the last segment does not have right
            margin. For example, if input sequence is tokenized into characters, ``max_seq_length=5``,
            ``step=1``, and ``margin=1``, then query "hello" will be tokenized into segments
            ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'e', 'l', 'l', '[SEP]'],
            ['[CLS]', 'l', 'l', 'o', '[SEP]']]``. These segments are passed to the model. Before final predictions
            computation, margins are removed. In the next list, subtokens which logits are not used for final
            predictions computation are marked with asterisk: ``[['[CLS]'*, 'h', 'e', 'l'*, '[SEP]'*],
            ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]``.

    Returns:
        all_input_ids: list of input ids of all segments
        all_segment_ids: token type ids of all segments
        all_input_mask: attention mask to use for BERT model
        all_subtokens_mask: masks out all subwords besides the first one
        all_quantities_of_preceding_words: number of words in query preceding a segment. Used for joining
            predictions from overlapping segments.
        all_query_ids: index of a query to which segment belongs
        all_is_first: is segment first segment in a query
        all_is_last: is segment last segment in a query
    """
    st = []
    stm = []
    sent_lengths = []
    for i, query in enumerate(queries):
        subtokens, subtokens_mask = _get_subtokens_and_subtokens_mask(query, tokenizer)
        sent_lengths.append(len(subtokens))
        st.append(subtokens)
        stm.append(subtokens_mask)
    _check_max_seq_length_and_margin_and_step(max_seq_length, margin, step)
    if max_seq_length > max(sent_lengths) + 2:
        max_seq_length = max(sent_lengths) + 2
        # If `max_seq_length` is greater than maximum length of input query, parameters ``margin`` and ``step`` are
        # not used will not be used.
        step = 1
        # Maximum number of word subtokens in segment. The first and the last tokens in segment are CLS and EOS
        length = max_seq_length - 2
    else:
        # Maximum number of word subtokens in segment. The first and the last tokens in segment are CLS and EOS
        length = max_seq_length - 2
        step = min(length - margin * 2, step)
    logging.info(f'Max length: {max_seq_length}')
    get_stats(sent_lengths)
    all_input_ids, all_segment_ids, all_subtokens_mask, all_input_mask, all_input_mask = [], [], [], [], []
    all_quantities_of_preceding_words, all_query_ids, all_is_first, all_is_last = [], [], [], []
    for q_i, query_st in enumerate(st):
        q_inp_ids, q_segment_ids, q_subtokens_mask, q_inp_mask, q_quantities_of_preceding_words = [], [], [], [], []
        for i in range(0, max(len(query_st), length) - length + step, step):
            subtokens = [tokenizer.cls_token] + query_st[i : i + length] + [tokenizer.sep_token]
            q_inp_ids.append(tokenizer.tokens_to_ids(subtokens))
            q_segment_ids.append([0] * len(subtokens))
            q_subtokens_mask.append([0] + stm[q_i][i : i + length] + [0])
            q_inp_mask.append([1] * len(subtokens))
            q_quantities_of_preceding_words.append(np.count_nonzero(stm[q_i][:i]))
        all_input_ids.append(q_inp_ids)
        all_segment_ids.append(q_segment_ids)
        all_subtokens_mask.append(q_subtokens_mask)
        all_input_mask.append(q_inp_mask)
        all_quantities_of_preceding_words.append(q_quantities_of_preceding_words)
        all_query_ids.append([q_i] * len(q_inp_ids))
        all_is_first.append([True] + [False] * (len(q_inp_ids) - 1))
        all_is_last.append([False] * (len(q_inp_ids) - 1) + [True])
    return (
        list(itertools.chain(*all_input_ids)),
        list(itertools.chain(*all_segment_ids)),
        list(itertools.chain(*all_input_mask)),
        list(itertools.chain(*all_subtokens_mask)),
        list(itertools.chain(*all_quantities_of_preceding_words)),
        list(itertools.chain(*all_query_ids)),
        list(itertools.chain(*all_is_first)),
        list(itertools.chain(*all_is_last)),
    )


class BertPunctuationCapitalizationInferDataset(Dataset):
    """
    Creates dataset to use during inference for punctuation and capitalization tasks with a pretrained model.
    For dataset to use during training with labels, see BertPunctuationCapitalizationDataset.

    Parameters ``max_seq_length``, ``step``, ``margin`` are for controlling the way queries are split into segments
    which then processed by the model. Parameter ``max_seq_length`` is a length of a segment after tokenization
    including special tokens [CLS] in the beginning and [SEP] in the end of a segment. Parameter ``step`` is shift
    between consequent segments. Parameter ``margin`` is used to exclude negative effect of subtokens near
    borders of segments which have only one side context.

    Args:
        queries: list of sequences.
        tokenizer: such as AutoTokenizer
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        step: relative shift of consequent segments into which long queries are split. Long queries are split into
            segments which can overlap. Parameter ``step`` controls such overlapping. Imagine that queries are
            tokenized into characters, ``max_seq_length=5``, and ``step=2``. In such a case query "hello" is
            tokenized into segments ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]``.
        margin: number of subtokens in the beginning and the end of segments which are not used for prediction
            computation. The first segment does not have left margin and the last segment does not have right
            margin. For example, if input sequence is tokenized into characters, ``max_seq_length=5``,
            ``step=1``, and ``margin=1``, then query "hello" will be tokenized into segments
            ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'e', 'l', 'l', '[SEP]'],
            ['[CLS]', 'l', 'l', 'o', '[SEP]']]``. These segments are passed to the model. Before final predictions
            computation, margins are removed. In the next list, subtokens which logits are not used for final
            predictions computation are marked with asterisk: ``[['[CLS]'*, 'h', 'e', 'l'*, '[SEP]'*],
            ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]``.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.

        input_ids: ids of word subtokens encoded using tokenizer
        segment_ids: an array of zeros
        input_mask: attention mask. Zeros if input is padding.
        subtoken_mask: a mask used for retrieving predictions for words. An element equals ``1`` if corresponding
            token is the first token in some word and zero otherwise. For example, if input query
            "language processing" is tokenized into ["[CLS]", "language", "process", "ing", "SEP"], then
            ``subtokens_mask`` will be [0, 1, 1, 0, 0].
        quantities_of_preceding_words: number of words preceding a segment in a query. It is used for uniting
            predictions from different segments if such segments overlap. For example, if query "hello john" is
            tokenized into segments ``[['hell', 'o'], ['john']]``, then ``quantities_of_preceding_words=[0, 1]``.
        query_ids: ids of queries to which segments belong. For example, if ``queries=["foo", "bar"]`` are
            segmented into ``[[['[CLS]', 'f', 'o', '[SEP]'], ['[CLS]', 'o', 'o', '[SEP]']],
            [['[CLS]', 'b', 'a', '[SEP]'], ['[CLS]', 'a', 'r', '[SEP]']]]``, then for batch
            [['[CLS]', 'o', 'o', '[SEP]'], ['[CLS]', 'b', 'a', '[SEP]'], ['[CLS]', 'a', 'r', '[SEP]']]
            ``query_ids=[0, 1, 1]``.
        is_first: is segment the first segment in query. The left margin of the first segment in a query is not
            removed and this parameter is used to identify first segments.
        is_last: is segment the last segment in query. The right margin of the last segment in a query is not
            removed and this parameter is used to identify last segments.

        """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'subtokens_mask': NeuralType(('B', 'T'), MaskType()),
            'quantities_of_preceding_words': NeuralType(('B',), Index()),
            'query_ids': NeuralType(('B',), Index()),
            'is_first': NeuralType(('B',), BoolType()),
            'is_last': NeuralType(('B',), BoolType()),
        }

    def __init__(
        self, queries: List[str], tokenizer: TokenizerSpec, max_seq_length: int = 128, step: int = 32, margin: int = 16
    ):
        features = get_features_infer(
            queries=queries, max_seq_length=max_seq_length, tokenizer=tokenizer, step=step, margin=margin
        )
        self.all_input_ids: List[List[int]] = features[0]
        self.all_segment_ids: List[List[int]] = features[1]
        self.all_input_mask: List[List[int]] = features[2]
        self.all_subtokens_mask: List[List[int]] = features[3]
        self.all_quantities_of_preceding_words: List[int] = features[4]
        self.all_query_ids: List[int] = features[5]
        self.all_is_first: List[bool] = features[6]
        self.all_is_last: List[bool] = features[7]

    def __len__(self) -> int:
        return len(self.all_input_ids)

    def collate_fn(
        self, batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, bool, bool]]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int], Tuple[int], Tuple[bool], Tuple[bool]
    ]:
        inp_ids, segment_ids, inp_mask, st_mask, n_preceding, query_ids, is_first, is_last = zip(*batch)
        return (
            pad_sequence([torch.tensor(x) for x in inp_ids], batch_first=True, padding_value=0),
            pad_sequence([torch.tensor(x) for x in segment_ids], batch_first=True, padding_value=0),
            pad_sequence([torch.tensor(x) for x in inp_mask], batch_first=True, padding_value=0),
            pad_sequence([torch.tensor(x) for x in st_mask], batch_first=True, padding_value=0),
            n_preceding,
            query_ids,
            is_first,
            is_last,
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, bool, bool]:
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.float32),
            np.array(self.all_subtokens_mask[idx]),
            self.all_quantities_of_preceding_words[idx],
            self.all_query_ids[idx],
            self.all_is_first[idx],
            self.all_is_last[idx],
        )
