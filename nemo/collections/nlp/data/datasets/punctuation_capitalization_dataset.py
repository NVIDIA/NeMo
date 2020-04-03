# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

__all__ = ['BertPunctuationCapitalizationDataset', 'BertPunctuationCapitalizationInferDataset']

import itertools
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils import get_label_stats, get_stats


def get_features(
    queries,
    max_seq_length,
    tokenizer,
    punct_label_ids=None,
    capit_label_ids=None,
    pad_label='O',
    punct_labels_lines=None,
    capit_labels_lines=None,
    ignore_extra_tokens=False,
    ignore_start_end=False,
):
    """
    Args:
    queries (list of str): text sequences
    max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
    tokenizer (TokenizerSpec): such as NemoBertTokenizer
    pad_label (str): pad value use for labels.
        by default, it's the neutral label.
    punct_label_ids (dict): dict to map punctuation labels to label ids.
        Starts with pad_label->0 and then increases in alphabetical order.
        Required for training and evaluation, not needed for inference.
    capit_label_ids (dict): dict to map labels to label ids. Starts
        with pad_label->0 and then increases in alphabetical order.
        Required for training and evaluation, not needed for inference.
    punct_labels (list of str): list of labels for every word in a sequence
    capit_labels (list of str): list of labels for every word in a sequence
    ignore_extra_tokens (bool): whether to ignore extra tokens in
        the loss_mask,
    ignore_start_end (bool): whether to ignore bos and eos tokens in
        the loss_mask
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
        all_loss_mask,
        all_subtokens_mask,
        punct_all_labels,
        capit_all_labels,
        punct_label_ids,
        capit_label_ids,
    )


class BertPunctuationCapitalizationDataset(Dataset):
    """
    Creates dataset to use during training for token classification
    tasks with a pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    For dataset to use during inference without labels, see
    BertPunctuationCapitalizationInferDataset.

    Args:
        text_file (str): file to sequences, each line should a sentence,
            No header.
        label_file (str): file to labels, each line corresponds to
            word labels for a sentence in the text_file. No header.
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as NemoBertTokenizer
        num_samples (int): number of samples you want to use for the dataset.
            If -1, use all dataset. Useful for testing.
        pad_label (str): pad value use for labels.
            by default, it's the neutral label.
        punct_label_ids and capit_label_ids (dict):
            dict to map labels to label ids.
            Starts with pad_label->0 and then increases in alphabetical order
            For dev set use label_ids generated during training to support
            cases when not all labels are present in the dev set.
            For training set label_ids should be None.
        ignore_extra_tokens (bool): whether to ignore extra tokens in
            the loss_mask,
        ignore_start_end (bool): whether to ignore bos and eos tokens in
            the loss_mask
    """

    def __init__(
        self,
        text_file,
        label_file,
        max_seq_length,
        tokenizer,
        num_samples=-1,
        pad_label='O',
        punct_label_ids=None,
        capit_label_ids=None,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
    ):

        if use_cache:
            # Cache features
            data_dir = os.path.dirname(text_file)
            filename = os.path.basename(text_file)

            if not filename.endswith('.txt'):
                raise ValueError("{text_file} should have extension .txt")

            filename = filename[:-4]
            tokenizer_type = type(tokenizer.tokenizer).__name__
            vocab_size = getattr(tokenizer, "vocab_size", 0)
            features_pkl = os.path.join(
                data_dir, "cached_{}_{}_{}_{}".format(filename, tokenizer_type, str(max_seq_length), str(vocab_size)),
            )

        if use_cache and os.path.exists(features_pkl):
            # If text_file was already processed, load from pickle
            features = pickle.load(open(features_pkl, 'rb'))
            logging.info(f'features restored from {features_pkl}')
        else:
            if num_samples == 0:
                raise ValueError("num_samples has to be positive", num_samples)

            with open(text_file, 'r') as f:
                text_lines = f.readlines()

            # Collect all possible labels
            punct_unique_labels = set([])
            capit_unique_labels = set([])
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

            if use_cache:
                pickle.dump(features, open(features_pkl, "wb"))
                logging.info(f'features saved to {features_pkl}')

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]
        self.punct_all_labels = features[5]
        self.capit_all_labels = features[6]
        self.punct_label_ids = features[7]
        self.capit_label_ids = features[8]

        # save label_ids
        def get_stats_and_save(all_labels, label_ids, name):
            infold = text_file[: text_file.rfind('/')]
            merged_labels = itertools.chain.from_iterable(all_labels)
            logging.info('Three most popular labels')
            _, label_frequencies, _ = get_label_stats(merged_labels, infold + '/label_count_' + name + '.tsv')

            out = open(os.path.join(infold, name + '_label_ids.csv'), 'w')
            labels, _ = zip(*sorted(label_ids.items(), key=lambda x: x[1]))
            out.write('\n'.join(labels))
            logging.info(f'Labels: {label_ids}')
            logging.info(f'Labels mapping saved to : {out.name}')

            return label_frequencies

        self.punct_label_frequencies = get_stats_and_save(self.punct_all_labels, self.punct_label_ids, 'punct')
        self.capit_label_frequencies = get_stats_and_save(self.capit_all_labels, self.capit_label_ids, 'capit')

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.long),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
            np.array(self.punct_all_labels[idx]),
            np.array(self.capit_all_labels[idx]),
        )


class BertPunctuationCapitalizationInferDataset(Dataset):
    """
    Creates dataset to use during inference for token classification
    tasks with a pretrained model.

    Converts from raw data to an instance that can be used by
    NMDataLayer.

    For dataset to use during training with labels, see
    BertPunctuationCapitalizationDataset.

    Args:
        queries (list): list of queries to run inference on
        max_seq_length (int): max sequence length minus 2 for [CLS] and [SEP]
        tokenizer (Tokenizer): such as NemoBertTokenizer
    """

    def __init__(self, queries, max_seq_length, tokenizer):
        features = get_features(queries, max_seq_length, tokenizer)

        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_loss_mask = features[3]
        self.all_subtokens_mask = features[4]

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.float32),
            np.array(self.all_loss_mask[idx]),
            np.array(self.all_subtokens_mask[idx]),
        )
