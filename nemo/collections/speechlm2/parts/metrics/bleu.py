from collections import defaultdict

import sacrebleu
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.utils import logging


class BLEU:
    """
    Computes BLEU scores on text predictions.
    By default, uses Whisper's EnglishTextNormalizer on hypotheses and references.
    """

    def __init__(self, normalize: bool = True, normalizer=None, verbose: bool = True):
        self.verbose = verbose
        if normalize:
            if normalizer is None:
                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity

        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def reset(self):
        return self

    def update(self, name: str, refs: list[str], hyps: list[str]) -> None:
        for ref, hyp in zip(refs, hyps):
            self._refs[name].append([self.normalizer(ref)])
            self._hyps[name].append(self.normalizer(hyp))
            if self.verbose:
                asrb = sacrebleu.sentence_bleu(hyp, [ref]).score
                logging.info(f"[REF]\t{ref}\n[HYP]\t{hyp} [{asrb:.2f}]")

    def compute(self) -> dict[str, torch.Tensor]:
        corpus_metric = {}
        for name in self._refs.keys():
            metric = torch.tensor(sacrebleu.corpus_bleu(self._hyps[name], self._refs[name]).score)
            corpus_metric[f"txt_bleu_{name}"] = metric
        corpus_metric["txt_bleu"] = torch.stack(list(corpus_metric.values())).mean()
        self._refs.clear()
        self._hyps.clear()
        return corpus_metric


def _identity(x):
    return x
