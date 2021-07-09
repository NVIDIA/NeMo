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

from math import ceil
from time import perf_counter
from typing import List

import numpy as np
import torch.nn as nn
from nltk import word_tokenize
from tqdm import tqdm
from transformers import *

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization import TextNormalizationTestDataset
from nemo.collections.nlp.models.duplex_text_normalization.utils import get_formatted_string
from nemo.utils import logging
from nemo.utils.decorators.experimental import experimental

__all__ = ['DuplexTextNormalizationModel']


@experimental
class DuplexTextNormalizationModel(nn.Module):
    """
    DuplexTextNormalizationModel is a wrapper class that can be used to
    encapsulate a trained tagger and a trained decoder. The class is intended
    to be used for inference only (e.g., for evaluation).
    """

    def __init__(self, tagger, decoder):
        super(DuplexTextNormalizationModel, self).__init__()

        self.tagger = tagger
        self.decoder = decoder

    def evaluate(
        self, dataset: TextNormalizationTestDataset, batch_size: int, errors_log_fp: str, verbose: bool = True
    ):
        """ Function for evaluating the performance of the model on a dataset

        Args:
            dataset: The dataset to be used for evaluation.
            batch_size: Batch size to use during inference. You can set it to be 1
                (no batching) if you want to measure the running time of the model
                per individual example (assuming requests are coming to the model one-by-one).
            errors_log_fp: Path to the file for logging the errors
            verbose: if true prints and logs various evaluation results

        Returns:
            results: A Dict containing the evaluation results (e.g., accuracy, running time)
        """
        results = {}
        error_f = open(errors_log_fp, 'w+')

        # Apply the model on the dataset
        all_run_times, all_dirs, all_inputs = [], [], []
        all_tag_preds, all_final_preds, all_targets = [], [], []
        nb_iters = int(ceil(len(dataset) / batch_size))
        for i in tqdm(range(nb_iters)):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_insts = dataset[start_idx:end_idx]
            batch_dirs, batch_inputs, batch_targets = zip(*batch_insts)
            # Inference and Running Time Measurement
            batch_start_time = perf_counter()
            batch_tag_preds, _, batch_final_preds = self._infer(batch_inputs, batch_dirs)
            batch_run_time = (perf_counter() - batch_start_time) * 1000  # milliseconds
            all_run_times.append(batch_run_time)
            # Update all_dirs, all_inputs, all_tag_preds, all_final_preds and all_targets
            all_dirs.extend(batch_dirs)
            all_inputs.extend(batch_inputs)
            all_tag_preds.extend(batch_tag_preds)
            all_final_preds.extend(batch_final_preds)
            all_targets.extend(batch_targets)

        # Metrics
        tn_error_ctx, itn_error_ctx = 0, 0
        for direction in constants.INST_DIRECTIONS:
            cur_dirs, cur_inputs, cur_tag_preds, cur_final_preds, cur_targets = [], [], [], [], []
            for dir, _input, tag_pred, final_pred, target in zip(
                all_dirs, all_inputs, all_tag_preds, all_final_preds, all_targets
            ):
                if dir == direction:
                    cur_dirs.append(dir)
                    cur_inputs.append(_input)
                    cur_tag_preds.append(tag_pred)
                    cur_final_preds.append(final_pred)
                    cur_targets.append(target)
            nb_instances = len(cur_final_preds)
            sent_accuracy = TextNormalizationTestDataset.compute_sent_accuracy(cur_final_preds, cur_targets, cur_dirs)
            if verbose:
                logging.info(f'\n============ Direction {direction} ============')
                logging.info(f'Sentence Accuracy: {sent_accuracy}')
                logging.info(f'nb_instances: {nb_instances}')
            # Update results
            results[direction] = {'sent_accuracy': sent_accuracy, 'nb_instances': nb_instances}
            # Write errors to log file
            for _input, tag_pred, final_pred, target in zip(cur_inputs, cur_tag_preds, cur_final_preds, cur_targets):
                if not TextNormalizationTestDataset.is_same(final_pred, target, direction):
                    if direction == constants.INST_BACKWARD:
                        error_f.write('Backward Problem (ITN)\n')
                        itn_error_ctx += 1
                    elif direction == constants.INST_FORWARD:
                        error_f.write('Forward Problem (TN)\n')
                        tn_error_ctx += 1
                    formatted_input_str = get_formatted_string(_input.split(' '))
                    formatted_tag_pred_str = get_formatted_string(tag_pred)
                    error_f.write(f'Original Input : {_input}\n')
                    error_f.write(f'Input          : {formatted_input_str}\n')
                    error_f.write(f'Predicted Tags : {formatted_tag_pred_str}\n')
                    error_f.write(f'Predicted      : {final_pred}\n')
                    error_f.write(f'Ground-Truth   : {target}\n')
                    error_f.write('\n')
            results['itn_error_ctx'] = itn_error_ctx
            results['tn_error_ctx'] = tn_error_ctx

        # Running Time
        avg_running_time = np.average(all_run_times) / batch_size  # in ms
        if verbose:
            logging.info(f'Average running time (normalized by batch size): {avg_running_time} ms')
        results['running_time'] = avg_running_time

        # Close log file
        error_f.close()

        return results

    # Functions for inference
    def _infer(self, sents: List[str], inst_directions: List[str]):
        """ Main function for Inference
        Args:
            sents: A list of input texts.
            inst_directions: A list of str where each str indicates the direction of the corresponding instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).

        Returns:
            tag_preds: A list of lists where each list contains the tag predictions from the tagger for an input text.
            output_spans: A list of lists where each list contains the decoded semiotic spans from the decoder for an input text.
            final_outputs: A list of str where each str is the final output text for an input text.
        """
        # Preprocessing
        sents = self.input_preprocessing(list(sents))

        # Tagging
        tag_preds, nb_spans, span_starts, span_ends = self.tagger._infer(sents, inst_directions)
        output_spans = self.decoder._infer(sents, nb_spans, span_starts, span_ends, inst_directions)

        # Preprare final outputs
        final_outputs = []
        for ix, (sent, tags) in enumerate(zip(sents, tag_preds)):
            cur_words, jx, span_idx = [], 0, 0
            cur_spans = output_spans[ix]
            while jx < len(sent):
                tag, word = tags[jx], sent[jx]
                if constants.SAME_TAG in tag:
                    cur_words.append(word)
                    jx += 1
                elif constants.PUNCT_TAG in tag:
                    jx += 1
                else:
                    jx += 1
                    cur_words.append(cur_spans[span_idx])
                    span_idx += 1
                    while jx < len(sent) and tags[jx] == constants.I_PREFIX + constants.TRANSFORM_TAG:
                        jx += 1
            cur_output_str = ' '.join(cur_words)
            cur_output_str = ' '.join(word_tokenize(cur_output_str))
            final_outputs.append(cur_output_str)
        return tag_preds, output_spans, final_outputs

    def input_preprocessing(self, sents):
        """ Function for preprocessing the input texts. The function first does
        some basic tokenization using nltk.word_tokenize() and then it processes
        Greek letters such as Δ or λ (if any).

        Args:
            sents: A list of input texts.

        Returns: A list of preprocessed input texts.
        """
        # Basic Preprocessing and Tokenization
        for ix, sent in enumerate(sents):
            sents[ix] = sents[ix].replace('+', ' plus ')
            sents[ix] = sents[ix].replace('=', ' equals ')
            sents[ix] = sents[ix].replace('@', ' at ')
            sents[ix] = sents[ix].replace('*', ' times ')
        sents = [word_tokenize(sent) for sent in sents]

        # Greek letters processing
        for ix, sent in enumerate(sents):
            for jx, tok in enumerate(sent):
                if tok in constants.GREEK_TO_SPOKEN:
                    sents[ix][jx] = constants.GREEK_TO_SPOKEN[tok]

        return sents
