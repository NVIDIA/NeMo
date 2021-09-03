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

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_stats
from nemo.collections.nlp.models import PunctuationCapitalizationModel
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, Index, MaskType, NeuralType
from nemo.core.neural_types.elements import BoolType
from nemo.utils import logging


"""
This script is for restoring punctuation and capitalization.

Usage example:

python punctuate_capitalize.py \
    --input_manifest <PATH_TO_INPUT_MANIFEST> \
    --output_manifest <PATH_TO_OUTPUT_MANIFEST>

<PATH_TO_INPUT_MANIFEST> is a path to NeMo ASR manifest. Usually it is an output of
    NeMo/examples/asr/transcribe_speech.py but can be a manifest with 'text' key. Alternatively you can use
    --input_text parameter for passing text for inference.
<PATH_TO_OUTPUT_MANIFEST> is a path to NeMo ASR manifest into which script output will be written. Alternatively
    you can use parameter --output_text.

For more details on this script usage look in argparse help.
"""


def get_args():
    default_model_parameter = "pretrained_name"
    default_model = "punctuation_en_bert"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="The script is for restoring punctuation and capitalization in text. Long strings are split into "
        "segments of length `--max_seq_length`. `--max_seq_length` is the length which includes [CLS] and [SEP] "
        "tokens. Parameter `--step` controls segments overlapping. `--step` is a distance between beginnings of "
        "consequent segments. Model outputs for tokens near the borders of tensors are less accurate and can be "
        "discarded before final predictions computation. Parameter `--margin` is number of discarded outputs near "
        "segments borders. Probabilities of tokens in overlapping parts of segments multiplied before selecting the "
        "best prediction.",
    )
    input_ = parser.add_mutually_exclusive_group(required=True)
    input_.add_argument(
        "--input_manifest",
        "-m",
        type=Path,
        help="Path to the file with NeMo manifest which needs punctuation and capitalization. If the first element "
        "of manifest contains key 'pred_text', 'pred_text' values are passed for tokenization. Otherwise 'text' "
        "values are passed for punctuation and capitalization. Exactly one parameter of `--input_manifest` and "
        "`--input_text` should be provided.",
    )
    input_.add_argument(
        "--input_text",
        "-t",
        type=Path,
        help="Path to file with text which needs punctuation and capitalization. Exactly one parameter of "
        "`--input_manifest` and `--input_text` should be provided.",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output_manifest",
        "-M",
        type=Path,
        help="Path to output NeMo manifest. Text with restored punctuation and capitalization will be saved in "
        "'pred_text' elements if 'pred_text' key is present in the input manifest. Otherwise text with restored "
        "punctuation and capitalization will be saved in 'text' elements. Exactly one parameter of `--output_manifest` "
        "and `--output_text` should be provided.",
    )
    output.add_argument(
        "--output_text",
        "-T",
        type=Path,
        help="Path to file with text with restored punctuation and capitalization. Exactly one parameter of "
        "`--output_manifest` and `--output_text` should be provided.",
    )
    model = parser.add_mutually_exclusive_group(required=False)
    model.add_argument(
        "--pretrained_name",
        "-p",
        help=f"The name of NGC pretrained model. No more than one of parameters `--pretrained_name`, `--model_path`"
        f"should be provided. If neither of parameters `--pretrained_name` and `--model_path`, then the script is run "
        f"with `--{default_model_parameter}={default_model}`.",
        choices=[m.pretrained_model_name for m in PunctuationCapitalizationModel.list_available_models()],
    )
    model.add_argument(
        "--model_path",
        "-P",
        type=Path,
        help=f"Path to .nemo checkpoint of punctuation and capitalization model. No more than one of parameters "
        f"`--pretrained_name` and `--model_path` should be provided. If neither of parameters `--pretrained_name` and "
        f"`--model_path`, then the script is run with `--{default_model_parameter}={default_model}`.",
    )
    parser.add_argument(
        "--max_seq_length",
        "-L",
        type=int,
        default=64,
        help="Length of segments into which queries are split. `--max_seq_length` includes [CLS] and [SEP] tokens.",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=8,
        help="Relative shift of consequent segments into which long queries are split. Long queries are split into "
        "segments which can overlap. Parameter `step` controls such overlapping. Imagine that queries are "
        "tokenized into characters, `max_seq_length=5`, and `step=2`. In such a case query 'hello' is tokenized "
        "into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`.",
    )
    parser.add_argument(
        "--margin",
        "-g",
        type=int,
        default=16,
        help="A number of subtokens in the beginning and the end of segments which output probabilities are not used "
        "for prediction computation. The first segment does not have left margin and the last segment does not have "
        "right margin. For example, if input sequence is tokenized into characters, `max_seq_length=5`, `step=1`, "
        "and `margin=1`, then query 'hello' will be tokenized into segments `[['[CLS]', 'h', 'e', 'l', '[SEP]'], "
        "['[CLS]', 'e', 'l', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]`. These segments are passed to the "
        "model. Before final predictions computation, margins are removed. In the next list, subtokens which logits "
        "are not used for final predictions computation are marked with asterisk: `[['[CLS]'*, 'h', 'e', 'l'*, "
        "'[SEP]'*], ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]`.",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=128, help="Number of segments which are processed simultaneously.",
    )
    args = parser.parse_args()
    if args.input_manifest is None and args.output_manifest is not None:
        parser.error("--output_manifest requires --input_manifest")
    if args.pretrained_name is None and args.model_path is None:
        setattr(args, default_model_parameter, default_model)
    for name in ["input_manifest", "input_text", "output_manifest", "output_text", "model_path"]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser())
    return args


def load_manifest(manifest):
    result = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            result.append(data)
    return result


def find_idx_of_first_nonzero(stm, start, max_, left):
    if left:
        end = start - max_
        step = -1
    else:
        end = start + max_
        step = 1
    result = None
    for i in range(start, end, step):
        if stm[i]:
            result = i
            break
    return result


def get_subtokens_and_subtokens_mask(query, tokenizer):
    words = query.strip().split()
    subtokens = []
    subtokens_mask = []
    for j, word in enumerate(words):
        word_tokens = tokenizer.text_to_tokens(word)
        subtokens.extend(word_tokens)
        subtokens_mask.append(1)
        subtokens_mask.extend([0] * (len(word_tokens) - 1))
    return subtokens, subtokens_mask


def check_max_seq_length_and_margin_and_step(max_seq_length, margin, step):
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
):
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
        all_input_ids: input ids for all tokens
        all_segment_ids: token type ids
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
        subtokens, subtokens_mask = get_subtokens_and_subtokens_mask(query, tokenizer)
        sent_lengths.append(len(subtokens))
        st.append(subtokens)
        stm.append(subtokens_mask)
    check_max_seq_length_and_margin_and_step(max_seq_length, margin, step)
    max_seq_length = min(max_seq_length, max(sent_lengths) + 2)
    logging.info(f'Max length: {max_seq_length}')
    # Maximum number of word subtokens in segment. The first and the last tokens in segment are CLS and EOS
    length = max_seq_length - 2
    step = min(length - margin * 2, step)
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


class BertPunctuationCapitalizationInferLongDataset(Dataset):
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
        self.all_input_ids = features[0]
        self.all_segment_ids = features[1]
        self.all_input_mask = features[2]
        self.all_subtokens_mask = features[3]
        self.all_quantities_of_preceding_words = features[4]
        self.all_query_ids = features[5]
        self.all_is_first = features[6]
        self.all_is_last = features[7]

    def __len__(self) -> int:
        return len(self.all_input_ids)

    def collate_fn(self, batch):
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

    def __getitem__(self, idx):
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


def setup_infer_dataloader(model, queries, batch_size, max_seq_length, step, margin):
    """
    Setup function for a infer data loader.

    Args:
        queries: lower cased text without punctuation
        batch_size: batch size to use during inference
        max_seq_length: maximum sequence length after tokenization
    Returns:
        A pytorch DataLoader.
    """
    if max_seq_length is None:
        max_seq_length = model._cfg.dataset.max_seq_length
    if step is None:
        step = model._cfg.dataset.step
    if margin is None:
        margin = model._cfg.dataset.margin

    dataset = BertPunctuationCapitalizationInferLongDataset(
        tokenizer=model.tokenizer, queries=queries, max_seq_length=max_seq_length, step=step, margin=margin
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=dataset.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=model._cfg.dataset.num_workers,
        pin_memory=model._cfg.dataset.pin_memory,
        drop_last=False,
    )


def move_acc_probs_to_token_preds(pred, acc_prob, number_of_probs_to_move):
    if number_of_probs_to_move > acc_prob.shape[0]:
        raise ValueError(
            f"Not enough accumulated probabilities. Number_of_probs_to_move={number_of_probs_to_move} "
            f"acc_prob.shape={acc_prob.shape}"
        )
    if number_of_probs_to_move > 0:
        pred = pred + list(np.argmax(acc_prob[:number_of_probs_to_move], axis=-1))
    acc_prob = acc_prob[number_of_probs_to_move:]
    return pred, acc_prob


def update_accumulated_probabilities(acc_prob, update):
    acc_prob *= update[: acc_prob.shape[0]]
    acc_prob = np.concatenate([acc_prob, update[acc_prob.shape[0] :]], axis=0)
    return acc_prob


def remove_margins(tensor, margin_size, keep_left, keep_right):
    if not keep_left:
        tensor = tensor[margin_size + 1:]  # remove left margin and CLS token
    if not keep_right:
        tensor = tensor[: tensor.shape[0] - margin_size - 1]  # remove right margin and SEP token
    return tensor


def apply_punct_capit_predictions(cfg, query, punct_preds, capit_preds):
    query = query.strip().split()
    assert len(query) == len(
        punct_preds
    ), f"len(query)={len(query)} len(punct_preds)={len(punct_preds)}, query[:30]={query[:30]}"
    assert len(query) == len(
        capit_preds
    ), f"len(query)={len(query)} len(capit_preds)={len(capit_preds)}, query[:30]={query[:30]}"
    punct_ids_to_labels = {v: k for k, v in cfg.punct_label_ids.items()}
    capit_ids_to_labels = {v: k for k, v in cfg.capit_label_ids.items()}
    query_with_punct_and_capit = ''
    for j, word in enumerate(query):
        punct_label = punct_ids_to_labels[punct_preds[j]]
        capit_label = capit_ids_to_labels[capit_preds[j]]

        if capit_label != cfg.dataset.pad_label:
            word = word.capitalize()
        query_with_punct_and_capit += word
        if punct_label != cfg.dataset.pad_label:
            query_with_punct_and_capit += punct_label
        query_with_punct_and_capit += ' '
    return query_with_punct_and_capit[:-1]


def transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
    punct_logits, capit_logits, subtokens_mask, start_word_ids, margin, is_first, is_last, query_ids
):
    new_start_word_ids = list(start_word_ids)
    subtokens_mask = subtokens_mask > 0.5
    b_punct_probs, b_capit_probs = [], []
    for i, (first, last, q_i, pl, cl, stm) in enumerate(
            zip(is_first, is_last, query_ids, punct_logits, capit_logits, subtokens_mask)
    ):
        if not first:
            new_start_word_ids[i] += torch.count_nonzero(stm[: margin + 1]).numpy()  # + 1 is for [CLS] token
        stm = remove_margins(stm, margin, keep_left=first, keep_right=last)
        for b_probs, logits in [(b_punct_probs, pl), (b_capit_probs, cl)]:
            p = torch.nn.functional.softmax(
                remove_margins(logits, margin, keep_left=first, keep_right=last)[stm], dim=-1,
            )
            b_probs.append(p.detach().cpu().numpy())
    return b_punct_probs, b_capit_probs, new_start_word_ids


def add_punctuation_capitalization(
    model: PunctuationCapitalizationModel,
    queries: List[str],
    batch_size: int = None,
    max_seq_length: int = 64,
    step: int = 8,
    margin: int = 16
):
    """
    Adds punctuation and capitalization to the queries. Use this method for inference.

    Parameters ``max_seq_length``, ``step``, ``margin`` are for controlling the way queries are split into segments
    which then processed by the model. Parameter ``max_seq_length`` is a length of a segment after tokenization
    including special tokens [CLS] in the beginning and [SEP] in the end of a segment. Parameter ``step`` is shift
    between consequent segments. Parameter ``margin`` is used to exclude negative effect of subtokens near
    borders of segments which have only one side context.

    If segments overlap, probabilities of overlapping predictions are multiplied and then the label with
    corresponding to the maximum probability is selected.

    Args:
        queries: lower cased text without punctuation
        batch_size: batch size to use during inference
        max_seq_length: maximum sequence length of segment after tokenization.
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
    Returns:
        result: text with added capitalization and punctuation ``max_seq_length`` equals 5, ``step`` equals 2, and

    """
    if len(queries) == 0:
        return []
    if batch_size is None:
        batch_size = len(queries)
        logging.info(f'Using batch size {batch_size} for inference')
    result = []
    mode = model.training
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model.eval()
        model = model.to(d)
        infer_datalayer = setup_infer_dataloader(model, queries, batch_size, max_seq_length, step, margin)

        # Predicted labels for queries. List of labels for every query
        all_punct_preds, all_capit_preds = [[] for _ in queries], [[] for _ in queries]
        # Accumulated probabilities (or product of probabilities acquired from different segments) of punctuation
        # and capitalization. Probabilities for words in query are extracted using `subtokens_mask`.Probabilities
        # for newly processed words are appended to the accumulated probabilities. If probabilities for a word are
        # already present in `acc_probs`, old probabilities are replaced with multiplication of old probabilities
        # and probabilities acquired from new segment. Segments are processed in the order they are present in an
        # input query. When all segments with a word are processed, the label with highest probability is chosen
        # and appended to an appropriate list in `all_preds`. After adding prediction to `all_preds`,
        # probabilities for a word are removed from `acc_probs`
        acc_punct_probs, acc_capit_probs = [None for _ in queries], [None for _ in queries]
        for batch_i, batch in enumerate(infer_datalayer):
            inp_ids, inp_type_ids, inp_mask, subtokens_mask, start_word_ids, query_ids, is_first, is_last = batch
            punct_logits, capit_logits = model.forward(
                input_ids=inp_ids.to(d), token_type_ids=inp_type_ids.to(d), attention_mask=inp_mask.to(d),
            )
            _res = transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
                punct_logits, capit_logits, subtokens_mask, start_word_ids, margin, is_first, is_last, query_ids
            )
            punct_probs, capit_probs, start_word_ids = _res
            for i, (q_i, start_word_id, bpp_i, bcp_i) in enumerate(
                    zip(query_ids, start_word_ids, punct_probs, capit_probs)
            ):
                for all_preds, acc_probs, b_probs_i in [
                    (all_punct_preds, acc_punct_probs, bpp_i),
                    (all_capit_preds, acc_capit_probs, bcp_i),
                ]:
                    if acc_probs[q_i] is None:
                        acc_probs[q_i] = b_probs_i
                    else:
                        all_preds[q_i], acc_probs[q_i] = move_acc_probs_to_token_preds(
                            all_preds[q_i], acc_probs[q_i], start_word_id - len(all_preds[q_i]),
                        )
                        acc_probs[q_i] = update_accumulated_probabilities(acc_probs[q_i], b_probs_i)
        for all_preds, acc_probs in [(all_punct_preds, acc_punct_probs), (all_capit_preds, acc_capit_probs)]:
            for q_i, (pred, prob) in enumerate(zip(all_preds, acc_probs)):
                if prob is not None:
                    all_preds[q_i], acc_probs[q_i] = move_acc_probs_to_token_preds(pred, prob, len(prob))
        for i, query in enumerate(queries):
            result.append(apply_punct_capit_predictions(model._cfg, query, all_punct_preds[i], all_capit_preds[i]))
    finally:
        # set mode back to its original value
        model.train(mode=mode)
    return result


def main():
    args = get_args()
    if args.pretrained_name is None:
        model = PunctuationCapitalizationModel.restore_from(args.model_path)
    else:
        model = PunctuationCapitalizationModel.from_pretrained(args.pretrained_name)
    if args.input_manifest is None:
        texts = []
        with args.input_text.open() as f:
            texts.append(f.readline().strip())
    else:
        manifest = load_manifest(args.input_manifest)
        text_key = "pred_text" if "pred_text" in manifest[0] else "text"
        texts = []
        for item in manifest:
            texts.append(item[text_key])
    processed_texts = add_punctuation_capitalization(
        model,
        texts,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        step=args.step,
        margin=args.margin,
    )
    if args.output_manifest is None:
        with args.output_text.open('w') as f:
            for t in processed_texts:
                f.write(t + '\n')
    else:
        with args.output_manifest.open('w') as f:
            for item, t in zip(manifest, processed_texts):
                item[text_key] = t
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()
