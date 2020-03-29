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

import os
import shutil
import re
from collections import defaultdict

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import DATABASE_EXISTS_TMP, if_exist, write_files


# put training files in convenient place for conversion to our format
def copy_input_files(infold):
    our_infold = infold+"/dataset"

    if os.path.exists(our_infold+"/trainset") and os.path.exists(our_infold+"/testset"):
        logging.info("Input folders exists")
        return

    logging.info(f"Copying files to input folder: {our_infold}")
    os.makedirs(our_infold, exist_ok=True)

    old_infold = infold + '/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation/KFold_1'
    if not os.path.exists(our_infold + "/trainset"):
        shutil.copytree(old_infold+'/trainset', our_infold+'/trainset')

    if not os.path.exists(our_infold + "/testset"):
        shutil.copytree(old_infold + '/testset/csv', our_infold + '/testset')


# prepare full data set  with all 25K queries from raw initial file
def create_big_dataset(infold):
    MIN_NUM_QUERIES = 25
    TRAIN_TRESHOLD = 0.8
    HEADER = ['answerid;scenario;intent;answer_annotation;answer_from_anno;answer_from_user']
    our_infold = infold + "/full_dataset"

    if os.path.exists(our_infold + "/trainset") and os.path.exists(our_infold + "/testset"):
        logging.info("Full dataset folder exists")
        return

    logging.info(f"Preparing full dataset folder: {our_infold}")
    os.makedirs(our_infold, exist_ok=True)

    query_dict = defaultdict(list)
    input_file = infold + '/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv'
    lines = open(input_file, 'r').readlines()
    for line in lines[1:]:
        parts = line.split(";")
        intent = parts[2] + "_" + parts[3]
        parts[5] = parts[5].replace("currency_name:", "currency_name :")
        parts[5] = re.sub("\s\s+", " ", parts[5])
        query = parts[1]+";"+parts[2]+";"+parts[3]+";\""+parts[5]+"\";\""+parts[8]+"\";\""+parts[9]+"\""
        query_dict[intent].append(query)

    os.makedirs(our_infold + "/trainset", exist_ok=True)
    os.makedirs(our_infold + "/testset", exist_ok=True)
    print(f'Number of intents: {len(query_dict)}')
    total_queries = 0
    training_queries = 0
    for intent in sorted(query_dict.keys()):
        print(f'{intent} - {len(query_dict[intent])}')

        # create files for intents that has minimal number of queries
        # divide 80/20 for train/test set files
        if len(query_dict[intent]) >= MIN_NUM_QUERIES:
            total_queries += len(query_dict[intent])
            query_dict[intent] = HEADER + query_dict[intent]
            train_num = int(TRAIN_TRESHOLD * len(query_dict[intent]))
            training_queries += train_num
            # creating training/testing files
            write_files(query_dict[intent][:train_num], f'{our_infold}/trainset/{intent}.csv')
            write_files(query_dict[intent][train_num:], f'{our_infold}/testset/{intent}.csv')

    print(f'Total queries: {total_queries}, Training: {training_queries} Testing: {total_queries-training_queries}')


# get list of intents from file names
def get_intents(infold):
    intents = [f[:-4] for f in os.listdir(infold)]
    intents.sort()
    print(f'Found {len(intents)} intents')
    return intents


# get list of queries with their corresponding intent number
def get_intent_queries(infold, intent_names, mode):
    intent_queries = ['sentence\tlabel\n']

    for index, intent in enumerate(intent_names):
        queries = open(f'{infold}/{mode}set/{intent}.csv', 'r').readlines()
        for query in queries[1:]:
            phrases = query.split(";")
            phrase = phrases[3][1:-1]
            words = phrase.split(" ")
            intent_query = ""
            for word in words:
                if word[0] == "[" or word[0] == ":":
                    continue
                elif word[-1] == "]":
                    intent_query += word[:-1] + " "
                else:
                    intent_query += word + " "

            intent_query = intent_query.strip() + "\t" + str(index)
            intent_queries.append(intent_query)

    return intent_queries


# find a lost of unique slot types in training and testing data
# we give a single slot type name both for starting and continuation tokes (not using B-, I- notation)
def get_slots(infold, modes):
    slots = set()

    for mode in modes:
        path = f'{infold}/{mode}set'
        for filename in os.listdir(path):
            lines = open(f'{path}/{filename}', 'r').readlines()
            for line in lines[1:]:
                query = line.split(";")[3]
                slot_phrases = re.findall('\[.*?\]', query)
                for slot_phrase in slot_phrases:
                    slot = slot_phrase.split(" : ")[0][1:]
                    slots.add(slot)

    slots = sorted(slots)
    slots.append("O")
    print(f'Found {len(slots)} slot types')
    return slots


# convert each word in a query to corresponding slot number
def get_slot_queries(infold, slot_dict, mode, intent_names):
    slot_queries = []
    outside_slot = len(slot_dict) - 1

    # keep the same order of files/queries as for intents
    for intent in intent_names:
        lines = open(f'{infold}/{mode}set/{intent}.csv', 'r').readlines()
        for line in lines[1:]:
            slot_query = ""
            query = line.split(";")[3]
            words = query.split(" ")
            current_slot = outside_slot
            for word in words:
                if word[0] == "[":
                    current_slot = slot_dict[word[1:]]
                elif word[0]  == ":":
                    continue
                else:
                    slot_query += str(current_slot) + " "
                    if word[-1] == ']':
                        current_slot = outside_slot

            slot_queries.append(slot_query.strip())

    return slot_queries


def process_assistant(infold, outfold, modes=['train', 'test'], use_full_dataset=True):
    """
    https://github.com/xliuhw/NLU-Evaluation-Data data set from 2018
    that includes about 25 thousand examples with 64 multi-domain intents and 54 entity types.
    It is already uncased, so there is no difference
    """
    # prepare big data set not with all 25K not generated in initial data
    create_big_dataset(infold)

    if if_exist(outfold, [f'{mode}_slots.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('Assistant', outfold))
        return outfold

    logging.info(f'Processing assistant commands dataset and store at {outfold}')
    os.makedirs(outfold, exist_ok=True)

    # copy train/test files to the convenient directory to work with
    copy_input_files(infold)

    if use_full_dataset:
        infold += "/full_dataset"
    else:
        infold += "/dataset"

    # get list of intents from train folder (test folder supposed to be the same)
    intent_names = get_intents(infold + "/trainset")
    write_files(intent_names, f'{outfold}/dict.intents.csv')

    # get all train and test queries with their intent
    for mode in modes:
        intent_queries = get_intent_queries(infold, intent_names, mode)
        write_files(intent_queries, f'{outfold}/{mode}.tsv')

    # get list of all unique slots in training and testing files
    slot_types = get_slots(infold, modes)
    # print(slot_types)
    write_files(slot_types, f'{outfold}/dict.slots.csv')

    # create files of slot queries
    slot_dict = {k: v for v, k in enumerate(slot_types)}
    for mode in modes:
        slot_queries = get_slot_queries(infold, slot_dict, mode, intent_names)
        write_files(slot_queries, f'{outfold}/{mode}_slots.tsv')

