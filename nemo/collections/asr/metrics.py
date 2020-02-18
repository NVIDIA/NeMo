# Copyright (c) 2019 NVIDIA Corporation
from typing import List, Optional

import torch


def __levenshtein(a: List, b: List) -> int:
    """Calculates the Levenshtein distance between a and b.
    The code was copied from: http://hetland.org/coding/python/levenshtein.py
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer


def classification_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, top_k: Optional[List[int]] = None
) -> List[float]:
    """
    Computes the top-k classification accuracy provided with
    un-normalized logits of a model and ground truth targets.
    If top_k is not provided, defaults to top_1 accuracy.
    If top_k is provided as a list, then the values are sorted
    in ascending order.
    Args:
        logits: Un-normalized logits of a model. Softmax will be
            applied to these logits prior to computation of accuracy.
        targets: Vector of integers which represent indices of class
            labels.
        top_k: Optional list of integers in the range [1, max_classes].
    Returns:
        A list of length `top_k`, where each value represents top_i
        accuracy (i in `top_k`).
    """
    if top_k is None:
        top_k = [1]
    max_k = max(top_k)

    with torch.no_grad():
        _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
        predictions = predictions.t()
        correct = predictions.eq(targets.view(1, -1)).expand_as(predictions)

        results = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().mean().to('cpu').numpy()
            results.append(correct_k)

    return results
