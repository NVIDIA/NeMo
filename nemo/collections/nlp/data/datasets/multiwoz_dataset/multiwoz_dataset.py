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

# =============================================================================
# Copyright 2019 Salesforce Research.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# =============================================================================

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/thu-coai/ConvLab-2
"""

import json
import os
import pickle
import random

from torch.utils.data import Dataset

from nemo.collections.nlp.data.datasets.multiwoz_dataset.multiwoz_slot_trans import REF_USR_DA
from nemo.utils import logging

__all__ = ['MultiWOZDataset', 'MultiWOZDataDesc']


class MultiWOZDataset(Dataset):
    """
    By default, use only vocab from training data
    Need to modify the code a little bit to create the vocab from all files
    """

    def __init__(self, data_dir, mode, domains, all_domains, vocab, gating_dict, slots, num_samples=-1, shuffle=False):

        logging.info(f'Processing {mode} data')
        self.data_dir = data_dir
        self.mode = mode
        self.gating_dict = gating_dict
        self.domains = domains
        self.all_domains = all_domains
        self.vocab = vocab
        self.slots = slots

        self.features, self.max_len = self.get_features(num_samples, shuffle)
        logging.info("Sample 0: " + str(self.features[0]))

    def get_features(self, num_samples, shuffle):
        if num_samples == 0:
            raise ValueError("num_samples has to be positive", num_samples)

        filename = f'{self.data_dir}/{self.mode}_dials.json'
        logging.info(f'Reading from {filename}')
        dialogs = json.load(open(filename, 'r'))

        domain_count = {}
        data = []
        max_resp_len, max_value_len = 0, 0

        for dialog_dict in dialogs:
            if num_samples > 0 and len(data) >= num_samples:
                break

            dialog_history = ""
            for domain in dialog_dict['domains']:
                if domain not in self.domains:
                    continue
                if domain not in domain_count:
                    domain_count[domain] = 0
                domain_count[domain] += 1

            for turn in dialog_dict['dialogue']:
                if num_samples > 0 and len(data) >= num_samples:
                    break

                turn_uttr = turn['system_transcript'] + ' ; ' + turn['transcript']
                turn_uttr_strip = turn_uttr.strip()
                dialog_history += turn["system_transcript"] + " ; " + turn["transcript"] + " ; "
                source_text = dialog_history.strip()

                turn_beliefs = fix_general_label_error_multiwoz(turn['belief_state'], self.slots)

                turn_belief_list = [f'{k}-{v}' for k, v in turn_beliefs.items()]

                gating_label, responses = [], []
                for slot in self.slots:
                    if slot in turn_beliefs:
                        responses.append(str(turn_beliefs[slot]))
                        if turn_beliefs[slot] == "dontcare":
                            gating_label.append(self.gating_dict["dontcare"])
                        elif turn_beliefs[slot] == "none":
                            gating_label.append(self.gating_dict["none"])
                        else:
                            gating_label.append(self.gating_dict["ptr"])
                    else:
                        responses.append("none")
                        gating_label.append(self.gating_dict["none"])

                sample = {
                    'ID': dialog_dict['dialogue_idx'],
                    'domains': dialog_dict['domains'],
                    'turn_domain': turn['domain'],
                    'turn_id': turn['turn_idx'],
                    'dialogue_history': source_text,
                    'turn_belief': turn_belief_list,
                    'gating_label': gating_label,
                    'turn_uttr': turn_uttr_strip,
                    'responses': responses,
                }

                sample['context_ids'] = self.vocab.tokens2ids(sample['dialogue_history'].split())
                sample['responses_ids'] = [
                    self.vocab.tokens2ids(y.split() + [self.vocab.eos]) for y in sample['responses']
                ]
                sample['turn_domain'] = self.all_domains[sample['turn_domain']]

                data.append(sample)

                resp_len = len(sample['dialogue_history'].split())
                max_resp_len = max(max_resp_len, resp_len)

        logging.info(f'Domain count{domain_count}')
        logging.info(f'Max response length{max_resp_len}')
        logging.info(f'Processing {len(data)} samples')

        if shuffle:
            logging.info(f'Shuffling samples.')
            random.shuffle(data)

        return data, max_resp_len

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        return {
            'dialog_id': item['ID'],
            'turn_id': item['turn_id'],
            'turn_belief': item['turn_belief'],
            'gating_label': item['gating_label'],
            'context_ids': item['context_ids'],
            'turn_domain': item['turn_domain'],
            'responses_ids': item['responses_ids'],
        }


class Vocab:
    """
    Vocab class for MultiWOZ dataset
    UNK_token = 0
    PAD_token = 1
    SOS_token = 3
    EOS_token = 2
    """

    def __init__(self):
        self.word2idx = {'UNK': 0, 'PAD': 1, 'EOS': 2, 'BOS': 3}
        self.idx2word = ['UNK', 'PAD', 'EOS', 'BOS']
        self.unk_id = self.word2idx['UNK']
        self.pad_id = self.word2idx['PAD']
        self.eos_id = self.word2idx['EOS']
        self.bos_id = self.word2idx['BOS']
        self.unk, self.pad, self.eos, self.bos = 'UNK', 'PAD', 'EOS', 'BOS'

    def __len__(self):
        return len(self.idx2word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def add_words(self, sent, level):
        """
        level == 'utterance': sent is a string
        level == 'slot': sent is a list
        level == 'belief': sent is a dictionary
        """
        if level == 'utterance':
            for word in sent.split():
                self.add_word(word)
        elif level == 'slot':
            for slot in sent:
                domain, info = slot.split('-')
                self.add_word(domain)
                for subslot in info.split(' '):
                    self.add_word(subslot)
        elif level == 'belief':
            for slot, value in sent.items():
                domain, info = slot.split('-')
                self.add_word(domain)
                for subslot in info.split(' '):
                    self.add_word(subslot)
                for val in value.split(' '):
                    self.add_word(val)

    def tokens2ids(self, tokens):
        """Converts list of tokens to list of ids."""
        return [self.word2idx[w] if w in self.word2idx else self.unk_id for w in tokens]


class MultiWOZDataDesc:
    """
    Processes MultiWOZ dataset, creates vocabulary file and list of slots.
    """

    def __init__(self, data_dir, domains={"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4}):
        logging.info(f'Processing MultiWOZ dataset')

        self.all_domains = {
            'attraction': 0,
            'restaurant': 1,
            'taxi': 2,
            'train': 3,
            'hotel': 4,
            'hospital': 5,
            'bus': 6,
            'police': 7,
        }
        self.gating_dict = {'ptr': 0, 'dontcare': 1, 'none': 2}

        self.data_dir = data_dir
        self.domains = domains
        self.vocab = Vocab()

        ontology_file = open(f'{self.data_dir}/ontology.json', 'r')
        self.ontology = json.load(ontology_file)

        # self.value_dict is ontology reformating in the following way, for example:
        # {'taxi-arrive by': list_of_values} -> {'taxi':{'arriveby': list_of_slot_values}}
        self.ontology_value_dict = json.load(open(f'{self.data_dir}/value_dict.json', 'r'))
        # detected dictionary of slot_name + (slot_name_domain) value pairs
        # from user dialogue acts
        self.det_dict = {}
        for domain, dic in REF_USR_DA.items():
            for key, value in dic.items():
                assert '-' not in key
                self.det_dict[key.lower()] = key + '-' + domain
                self.det_dict[value.lower()] = key + '-' + domain

        self.vocab_file = None
        self.slots = None

        self.get_slots()
        self.get_vocab()

    def get_vocab(self):
        self.vocab_file = f'{self.data_dir}/vocab.pkl'

        if os.path.exists(self.vocab_file):
            logging.info(f'Loading vocab from {self.data_dir}')
            self.vocab = pickle.load(open(self.vocab_file, 'rb'))
        else:
            self.create_vocab()

        logging.info(f'Vocab size {len(self.vocab)}')

    def get_slots(self):
        used_domains = [key for key in self.ontology if key.split('-')[0] in self.domains]
        self.slots = [k.replace(' ', '').lower() if 'book' not in k else k.lower() for k in used_domains]

    def create_vocab(self):
        self.vocab.add_words(self.slots, 'slot')

        filename = f'{self.data_dir}/train_dials.json'
        logging.info(f'Building vocab from {filename}')
        dialogs = json.load(open(filename, 'r'))

        max_value_len = 0

        for dialog_dict in dialogs:
            for turn in dialog_dict['dialogue']:
                self.vocab.add_words(turn['system_transcript'], 'utterance')
                self.vocab.add_words(turn['transcript'], 'utterance')

                turn_beliefs = fix_general_label_error_multiwoz(turn['belief_state'], self.slots)
                lengths = [len(turn_beliefs[slot]) for slot in self.slots if slot in turn_beliefs]
                lengths.append(max_value_len)
                max_value_len = max(lengths)

        logging.info(f'Saving vocab to {self.data_dir}')
        with open(self.vocab_file, 'wb') as handle:
            pickle.dump(self.vocab, handle)


def fix_general_label_error_multiwoz(labels, slots):
    label_dict = dict([label['slots'][0] for label in labels])
    GENERAL_TYPO = {
        # type
        "guesthouse": "guest house",
        "guesthouses": "guest house",
        "guest": "guest house",
        "mutiple sports": "multiple sports",
        "sports": "multiple sports",
        "mutliple sports": "multiple sports",
        "swimmingpool": "swimming pool",
        "concerthall": "concert hall",
        "concert": "concert hall",
        "pool": "swimming pool",
        "night club": "nightclub",
        "mus": "museum",
        "ol": "architecture",
        "colleges": "college",
        "coll": "college",
        "architectural": "architecture",
        "musuem": "museum",
        "churches": "church",
        # area
        "center": "centre",
        "center of town": "centre",
        "near city center": "centre",
        "in the north": "north",
        "cen": "centre",
        "east side": "east",
        "east area": "east",
        "west part of town": "west",
        "ce": "centre",
        "town center": "centre",
        "centre of cambridge": "centre",
        "city center": "centre",
        "the south": "south",
        "scentre": "centre",
        "town centre": "centre",
        "in town": "centre",
        "north part of town": "north",
        "centre of town": "centre",
        "cb30aq": "none",
        # price
        "mode": "moderate",
        "moderate -ly": "moderate",
        "mo": "moderate",
        # day
        "next friday": "friday",
        "monda": "monday",
        # parking
        "free parking": "free",
        # internet
        "free internet": "yes",
        # star
        "4 star": "4",
        "4 stars": "4",
        "0 star rarting": "none",
        # others
        "y": "yes",
        "any": "dontcare",
        "n": "no",
        "does not care": "dontcare",
        "not men": "none",
        "not": "none",
        "not mentioned": "none",
        '': "none",
        "not mendtioned": "none",
        "3 .": "3",
        "does not": "no",
        "fun": "none",
        "art": "none",
    }

    hotel_ranges = [
        "nigh",
        "moderate -ly priced",
        "bed and breakfast",
        "centre",
        "venetian",
        "intern",
        "a cheap -er hotel",
    ]
    locations = ["gastropub", "la raza", "galleria", "gallery", "science", "m"]
    detailed_hotels = ["hotel with free parking and free wifi", "4", "3 star hotel"]
    areas = ["stansted airport", "cambridge", "silver street"]
    attr_areas = ["norwich", "ely", "museum", "same area as hotel"]

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])

            # miss match slot and value
            if (
                (slot == "hotel-type" and label_dict[slot] in hotel_ranges)
                or (slot == "hotel-internet" and label_dict[slot] == "4")
                or (slot == "hotel-pricerange" and label_dict[slot] == "2")
                or (slot == "attraction-type" and label_dict[slot] in locations)
                or ("area" in slot and label_dict[slot] in ["moderate"])
                or ("day" in slot and label_dict[slot] == "t")
            ):
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in detailed_hotels:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no":
                    label_dict[slot] = "north"
                elif label_dict[slot] == "we":
                    label_dict[slot] = "west"
                elif label_dict[slot] == "cent":
                    label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we":
                    label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no":
                    label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if (slot == "restaurant-area" and label_dict[slot] in areas) or (
                slot == "attraction-area" and label_dict[slot] in attr_areas
            ):
                label_dict[slot] = "none"

    return label_dict
