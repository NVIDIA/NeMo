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

import torch.nn as nn

from transformers import *
from nltk import word_tokenize
from typing import List
from nemo.collections.nlp.data.text_normalization.constants import (
    I_PREFIX, SAME_TAG, PUNCT_TAG, TRANSFORM_TAG, GREEK_TO_SPOKEN
)

__all__ = ['NeuralTextNormalizationModel']

class NeuralTextNormalizationModel(nn.Module):
    """ NeuralTextNormalizationModel is a wrapper class that can be used to
    encapsulate a trained tagger and a trained decoder. The class is intended
    to be used for inference only (e.g., for evaluation).
    """

    def __init__(self, tagger, decoder):
        super(NeuralTextNormalizationModel, self).__init__()

        self.tagger = tagger
        self.decoder = decoder

    # Functions for inference
    def _infer(self, sents: List[str]):
        # Preprocessing
        sents = self.input_preprocessing(sents)

        # Tagging
        tag_preds, nb_spans, span_starts, span_ends = self.tagger._infer(sents)
        output_spans = self.decoder._infer(sents, nb_spans, span_starts, span_ends)

        # Preprare final outputs
        final_outputs = []
        for ix, (sent, tags) in enumerate(zip(sents, tag_preds)):
            cur_words, jx, span_idx = [], 0, 0
            cur_spans = output_spans[ix]
            while jx < len(sent):
                tag, word = tags[jx], sent[jx]
                if SAME_TAG in tag:
                    cur_words.append(word)
                    jx += 1
                elif PUNCT_TAG in tag:
                    jx += 1
                else:
                    jx += 1
                    cur_words.append(cur_spans[span_idx])
                    span_idx += 1
                    while jx < len(sent) and tags[jx] == I_PREFIX + TRANSFORM_TAG:
                        jx += 1
            cur_output_str = ' '.join(cur_words)
            cur_output_str = ' '.join(word_tokenize(cur_output_str))
            final_outputs.append(cur_output_str)
        return final_outputs

    def input_preprocessing(self, sents):
        # Basic Tokenization
        sents = [word_tokenize(sent) for sent in sents]

        # Greek letters processing
        for ix, sent in enumerate(sents):
            for jx, tok in enumerate(sent):
                if tok in GREEK_TO_SPOKEN:
                    sents[ix][jx] = GREEK_TO_SPOKEN[tok]

        return sents
