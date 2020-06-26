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

'''
LaserTagger preprocessor for converting the training, validation, and test files
to examples used in the LaserTagger main file.
'''

import os
from collections import deque

import torch
from official_lasertagger import bert_example, tagging_converter, utils

import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.utils import NemoArgParser


def parse_args():
    '''
    LaserTagger preprocessor argument parser
    '''
    parser = NemoArgParser(description="LaserTagger Preprocessor")
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
    parser.add_argument(
        '--pretrained_model_name',
        default='bert-base-cased',
        type=str,
        help='Name of the pre-trained model',
        choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
    )
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--save_path", default=None, help="Path to the save the preprocessed data.")

    return parser.parse_args()


def read_input_file(
    args, input_file, output_arbitrary_targets_for_infeasible_examples=False, save_tokens=False, infer=False
):
    '''Reads in Tab Separated Value file and converts to training/infernece-ready examples.

    Args:
        args: Parsed args returned by the parse_args().
        input_file: Path to the TSV input file.
        output_arbitrary_targets_for_infeasible_examples: Set this to True when preprocessing 
            the development set. Determines whether to output a TF example also for sources 
            that can not be converted to target via the available tagging operations. In these 
            cases, the target ids will correspond to the tag sequence KEEP-DELETE-KEEP-DELETE... 
            which should be very unlikely to be predicted by chance. This will be useful for 
            getting more accurate eval scores during training.
        save_tokens: To save tokens required in example.task, only needs to be True for testing.
        infer: Whether test files or not.

    Returns:
        examples: List of converted examples(features and Editing Tasks).
        saved_tokens: List of additional out-of-vocab special tokens in test files.
    '''

    label_map = utils.read_label_map(args.label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map), True
    )
    builder = bert_example.BertExampleBuilder(
        label_map, args.pretrained_model_name, args.max_seq_length, False, converter
    )
    examples = deque()
    for i, (sources, target) in enumerate(utils.yield_sources_and_targets(input_file)):
        if len(examples) % 1000 == 0:
            logging.info("{} examples processed.".format(len(examples)))
        example = builder.build_bert_example(
            sources, target, output_arbitrary_targets_for_infeasible_examples, save_tokens, infer
        )
        if example is None:
            continue
        examples.append(example)
    logging.info(f'Done. {len(examples)} examples converted.')
    return examples, builder.get_special_tokens_and_ids()


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_examples, _ = read_input_file(args, args.train_file, False)
    eval_examples, eval_special_tokens = read_input_file(args, args.eval_file, True)
    test_examples, test_special_tokens = read_input_file(args, args.test_file, False, save_tokens=True, infer=True)

    torch.save((train_examples), args.save_path + "/lt_train_examples.pkl")
    torch.save((eval_examples, eval_special_tokens), args.save_path + "/lt_eval_examples.pkl")
    torch.save((test_examples, test_special_tokens), args.save_path + "/lt_test_examples.pkl")
