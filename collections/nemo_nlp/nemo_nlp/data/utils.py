import os
import pickle

import numpy as np

from nemo.utils.exp_logging import get_logger

logger = get_logger('')


def dataset_to_ids(dataset, tokenizer, cache_ids=False, add_bos_eos=True):
    """
    Reads dataset from file line by line, tokenizes each line with tokenizer,
    and returns list of lists which corresponds to ids of tokenized strings.

    Args:
        dataset: path to dataset
        tokenizer: tokenizer to convert text into ids
        cache_ids: if True, ids are saved to disk as pickle file
            with similar name (e.g., data.txt --> data.txt.pkl)
        add_bos_eos: bool, whether to add <s> and </s> symbols (e.g., for NMT)
    Returns:
        ids: list of ids which correspond to tokenized strings of the dataset
    """

    cached_ids_dataset = dataset + str(".pkl")
    if os.path.isfile(cached_ids_dataset):
        logger.info("Loading cached tokenized dataset ...")
        ids = pickle.load(open(cached_ids_dataset, "rb"))
    else:
        logger.info("Tokenizing dataset ...")
        data = open(dataset, "rb").readlines()
        ids = []
        for sentence in data:
            sent_ids = tokenizer.text_to_ids(sentence.decode("utf-8"))
            if add_bos_eos:
                sent_ids = [tokenizer.bos_id()] + sent_ids + \
                           [tokenizer.eos_id()]
            ids.append(sent_ids)
        if cache_ids:
            logger.info("Caching tokenized dataset ...")
            pickle.dump(ids, open(cached_ids_dataset, "wb"))
    return ids


def clean_src_and_target(src_ids, tgt_ids, max_tokens=128, min_tokens=3,
                         max_tokens_diff=25, max_tokens_ratio=2.5):
    """
    Cleans source and target sentences to get rid of noisy data.
    Specifically, a pair of sentences is removed if
      -- either source or target is longer than *max_tokens*
      -- either source or target is shorter than *min_tokens*
      -- absolute difference between source and target is larger than
         *max_tokens_diff*
      -- one sentence is *max_tokens_ratio* times longer than the other
    """

    if len(src_ids) != len(tgt_ids):
        raise ValueError("Source and target corpora have different lengths!")
    src_ids_, tgt_ids_ = [], []
    for i in range(len(src_ids)):
        src_len, tgt_len = len(src_ids[i]), len(tgt_ids[i])
        if src_len > max_tokens or tgt_len > max_tokens or \
                src_len < min_tokens or tgt_len < min_tokens or \
                (src_ids[i] == tgt_ids[i]) or \
                np.abs(src_len - tgt_len) > max_tokens_diff:
            continue
        ratio = max(src_len - 2, 1) / max(tgt_len - 2, 1)
        if ratio > max_tokens_ratio or ratio < (1 / max_tokens_ratio):
            continue
        src_ids_.append(src_ids[i])
        tgt_ids_.append(tgt_ids[i])
    return src_ids_, tgt_ids_
