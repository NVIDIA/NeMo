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

import itertools
import pickle as pkl
import random
from argparse import ArgumentParser

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

# Info on these headers can be found here on the UMLS website https://www.ncbi.nlm.nih.gov/books/NBK9685/
# section 3.3.4 Table 1
HEADERS = [
    'CUI',
    'LAT',
    'TS',
    'LUI',
    'STT',
    'SUI',
    'ISPREF',
    'AUI',
    'SAUI',
    'SCUI',
    'SDUI',
    'SAB',
    'TTY',
    'CODE',
    'STR',
    'SRL',
    'SUPPRESS',
    'CVF',
]


def process_umls_training_dataset(data_path, train_save_name, val_save_name, max_pairs, train_split, headers):
    """
    Generates and saves UMLS self alignment pretraining train and validation data. Takes the raw .RRF UMLS 
    data file and creates different pair combinations for entities with the same CUI. Each row in the output
    will be formatted as 'CUI EntitySynonym1 EntitySynonym2' with each item in a row separated by tabs.
    Saves two .tsv output files, one for the train split and one for the validation split.
    Only data marked as English is added to the train and val splits. 

    Arguments:
        data_path (str): path to MRCONSO.RRF UMLS data file
        train_save_name (str): path to where training data will be saved
        val_save_name (str): path to where validation data will be saved
        max_pairs (int): max number of pairs for any one CUI added to the train 
                   or validation splits
        train_split (float): precentage of raw data to be added to train set split
        headers (list): column lables within MRCONSO.RRF
    """

    print("Loading training data file...")
    df = pd.read_table(data_path, names=headers, index_col=False, delimiter='|')
    train_file = open(train_save_name, 'w')
    val_file = open(val_save_name, 'w')

    cui = df["CUI"].iloc[0]
    names = []
    random.seed(2021)

    for idx in tqdm(range(len(df))):
        # Address incorrectly formatted data
        if type(df["STR"].iloc[idx]) != str or "|" in df["STR"].iloc[idx]:
            continue

        # Collect all english concept strings matching the current CUI
        if df["CUI"].iloc[idx] == cui and df["LAT"].iloc[idx] == "ENG":
            concept_string = df["STR"].iloc[idx]
            names.append(concept_string)

        else:
            # Pair off concept synonyms to make training and val sets
            pairs = list(itertools.combinations(names, 2))

            if len(pairs) == 0:
                # Not enough concepts gathered to make a pair
                cui = df["CUI"].iloc[idx]
                names = [df["STR"].iloc[idx]]
                continue

            # Removing leading C to convert label string to int
            cui = int(cui[1:])
            random.shuffle(pairs)

            # Keep up to max pairs number pairs for any one concept
            for pair in pairs[:max_pairs]:

                # Want concepts in train and val splits to be randomly selected and mutually exclusive
                add_to_train = random.random()

                if add_to_train <= train_split:
                    train_file.write(f'{cui}\t{pair[0]}\t{pair[1]}\n')
                else:
                    val_file.write(f'{cui}\t{pair[0]}\t{pair[1]}\n')

            # Switch to next concept
            cui = df["CUI"].iloc[idx]
            names = [df["STR"].iloc[idx]]

    train_file.close()
    val_file.close()
    print("Finished making training and validation data")


def process_umls_index_dataset(data_path, data_savename, id2string_savename, headers):
    """
    Generates data file needed to build a UMLS index and a hash table mapping each
    CUI to one canonical concept string. Takes the raw .RRF data file and saves 
    a .tsv indec concept file as well as the a .pkl file of cui to concept string 
    mappings. Only data marked as English is added to the index data file. 

    Arguments:
        data_path (str): path to MRCONSO.RRF UMLS data file
        data_savename (str): path to where .tsv index data will be saved
        id2string_savename (str): path to where .pkl cui to string mapping will
                                  be saved
        headers (list): column lables within MRCONSO.RRF
    """

    print("Loading index data file...")
    df = pd.read_table(data_path, names=headers, index_col=False, delimiter='|')
    id2string = {}

    with open(data_savename, "w") as outfile:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Address incorrectly formatted data
            if type(row["STR"]) != str or "|" in row["STR"]:
                continue

            cui = row["CUI"]
            sent = row["STR"]

            # Removing leading C to convert label string to int
            cui = int(cui[1:])

            # Only keeping english concepts
            if row["LAT"] == "ENG":
                outfile.write(f'{cui}\t{sent}\n')

                # Matching each cui to one canonical string represention
                if cui not in id2string and ":" not in sent:
                    id2string[cui] = sent

    outfile.close()
    pkl.dump(id2string, open(id2string_savename, "wb"))
    print("Finished saving index data and id to concept mapping")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Whether to process data for building an index")
    parser.add_argument("--project_dir", required=False, type=str, default=".")
    parser.add_argument("--cfg", required=False, type=str, default="conf/umls_medical_entity_linking_config.yaml")
    parser.add_argument(
        "--max_pairs", required=False, type=int, default=50, help="Max number of train pairs for a single concepts"
    )
    parser.add_argument(
        "--train_split", required=False, type=float, default=0.99, help="Precentage of data to add to train set"
    )

    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    cfg.project_dir = args.project_dir

    if args.index:
        process_umls_index_dataset(cfg.index.raw_data, cfg.index.index_ds.data_file, cfg.index.id_to_string, HEADERS)
    else:
        process_umls_training_dataset(
            cfg.model.raw_data,
            cfg.model.train_ds.data_file,
            cfg.model.validation_ds.data_file,
            args.max_pairs,
            args.train_split,
            HEADERS,
        )
