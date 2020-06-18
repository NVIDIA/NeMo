# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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

import argparse
import json
import os
from collections import OrderedDict, deque

import torch
from examples.nlp.lasertagger.official_lasertagger import bert_example, tagging_converter, utils

from nemo import logging
from nemo.collections.nlp.data.tokenizers import bert_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="LaserTagger Preprocessor")
    parser.add_argument(
        "--train_file", type=str, help="The training data file. Should be *.tsv",
    )
    parser.add_argument(
        "--eval_file", type=str, help="The evaluation data file. Should be *.tsv",
    )
    parser.add_argument(
        "--test_file", type=str, help="The test data file. Should be *.tsv",
    )

    parser.add_argument(
        "--label_map_file",
        type=str,
        help="Path to the label map file. Either a JSON file ending with '.json', \
				that maps each possible tag to an ID, or a text file that \
				has one tag per line.",
    )
    parser.add_argument("--vocab_file", default=None, help="Path to the vocab file.")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--save_path", default=None, help="Path to the save the preprocessed data.")

    return parser.parse_args()


def read_input_file(args, input_file, output_arbitrary_targets_for_infeasible_examples=False, save_tokens=True):

    label_map = utils.read_label_map(args.label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map), True
    )
    builder = bert_example.BertExampleBuilder(label_map, args.vocab_file, args.max_seq_length, False, converter)
    num_converted = 0
    examples = deque()
    for i, (sources, target) in enumerate(utils.yield_sources_and_targets(input_file)):
        logging.log_every_n(logging.INFO, f'{i} examples processed, {num_converted} converted.', 10000)
        example = builder.build_bert_example(
            sources, target, output_arbitrary_targets_for_infeasible_examples, save_tokens
        )
        if example is None:
            continue
        num_converted += 1
        examples.append(example)
    logging.info(f'Done. {num_converted} examples converted.')

    tokenizer = bert_tokenizer.NemoBertTokenizer(vocab_file=args.vocab_file, do_lower_case=False)
    return examples, num_converted, builder.get_special_tokens_and_ids()


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    processed = OrderedDict()
    train_examples, num_train_examples, _ = read_input_file(args, args.train_file, False, False)
    eval_examples, num_eval_examples, eval_special_tokens = read_input_file(args, args.eval_file, True, False)
    test_examples, num_test_examples, test_special_tokens = read_input_file(args, args.test_file, False, True)

    torch.save((train_examples, num_train_examples), args.save_path + "/lt_train_examples.pkl")
    torch.save((eval_examples, num_eval_examples, eval_special_tokens), args.save_path + "/lt_eval_examples.pkl")
    torch.save((test_examples, num_test_examples, test_special_tokens), args.save_path + "/lt_test_examples.pkl")
