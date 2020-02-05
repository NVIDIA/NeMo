#!/usr/bin/python

# =============================================================================
# Copyright 2019 NVIDIA. All Rights Reserved.
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
# Copyright 2019 Salesforce Research and Paweł Budzianowski.
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
Dataset: http://dialogue.mi.eng.cam.ac.uk/index.php/corpus/

Code based on:
https://github.com/jasonwu0731/trade-dst
https://github.com/budzianowski/multiwoz
"""

import argparse
import json
import os
import re
import shutil

from nemo.collections.nlp.data.datasets.datasets_utils import if_exist

parser = argparse.ArgumentParser(description='Process MultiWOZ dataset')
parser.add_argument("--data_dir", default='../../data/statetracking/MULTIWOZ2.1', type=str)
parser.add_argument("--out_dir", default='../../data/statetracking/multiwoz', type=str)
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(f"{args.data_dir} doesn't exist.")

DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
PHONE_NUM_TMPL = '\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})'
POSTCODE_TMPL = (
    '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?' + '[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})'
)

REPLACEMENTS = {}
with open('replacements.txt', 'r') as f:
    for line in f:
        word1, word2 = line.strip().split('\t')
        REPLACEMENTS[word1] = word2
REPLACEMENTS['-'] = ' '
REPLACEMENTS[';'] = ','
REPLACEMENTS['/'] = ' and '

DONT_CARES = set(['dont care', 'dontcare', "don't care", "do not care"])


def is_ascii(text):
    return all(ord(c) < 128 for c in text)


def normalize(text):
    text = text.lower().strip()

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove brackets
    text = re.sub(u"(\u2018|\u2019)", "'", text)  # weird unicode bug
    # add space around punctuations
    text = re.sub('(\D)([?.,!])', r'\1 \2 ', text)

    clean_tokens = []

    for token in text.split():
        token = token.strip()
        if not token:
            continue
        if token in REPLACEMENTS:
            clean_tokens.append(REPLACEMENTS[token])
        else:
            clean_tokens.append(token)

    text = ' '.join(clean_tokens)  # remove extra spaces
    text = re.sub('(\d) (\d)', r'\1\2', text)  # concatenate numbers

    return text


def get_goal(idx, log, goals, last_goal):
    if idx == 1:  # first system's response
        active_goals = get_summary_belief_state(log[idx]["metadata"], True)
        return active_goals[0] if len(active_goals) != 0 else goals[0]
    else:
        new_goals = get_new_goal(log[idx - 2]["metadata"], log[idx]["metadata"])
        return last_goal if not new_goals else new_goals[0]


def get_summary_belief_state(bstate, get_goal=False):
    """Based on the mturk annotations we form multi-domain belief state
    TODO: Figure out why this script has hotel-name but jason's script doesn't
    (see val_dialogs.json)
    """
    summary_bstate, summary_bvalue, active_domain = [], [], []
    for domain in DOMAINS:
        domain_active = False
        booking = []

        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                booking.append(int(len(bstate[domain]['book']['booked']) != 0))
            else:
                if bstate[domain]['book'][slot]:
                    booking.append(1)
                    curr_bvalue = [f"{domain}-book {slot.strip().lower()}", normalize(bstate[domain]['book'][slot])]
                    summary_bvalue.append(curr_bvalue)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book']:
                booking.append(0)
            if 'ticket' not in bstate[domain]['book']:  # TODO: possibly elif
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] in DONT_CARES:
                slot_enc[1] = 1
                summary_bvalue.append([f"{domain}-{slot.strip().lower()}", "dontcare"])
            elif bstate[domain]['semi'][slot]:
                curr_bvalue = [f"{domain}-{slot.strip().lower()}", normalize(bstate[domain]['semi'][slot])]
                summary_bvalue.append(curr_bvalue)
            if sum(slot_enc) > 0:
                domain_active = True
            summary_bstate += slot_enc

        if domain_active:  # quasi domain-tracker
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    assert len(summary_bstate) == 94
    if get_goal:
        return active_domain
    return summary_bstate, summary_bvalue


def get_new_goal(prev_turn, curr_turn):
    """ If multiple domains are updated between turns,
    return all of them
    """
    new_goals = []
    # Sometimes, metadata is an empty dictionary, bug?
    if not prev_turn or not curr_turn:
        return new_goals

    for domain in prev_turn:
        if curr_turn[domain] != prev_turn[domain]:
            new_goals.append(domain)
    return new_goals


def get_dialog_act(curr_dialog_acts, act_idx):
    """Given system dialogue acts fix automatic delexicalization."""
    acts = []
    if not act_idx in curr_dialog_acts:
        return acts

    turn = curr_dialog_acts[act_idx]

    if isinstance(turn, dict):  # it's annotated:
        for key in turn:
            key_acts = turn[key]
            key = key.strip().lower()
            if key.endswith('request'):
                for act in key_acts:
                    acts.append(act[0].lower())
            elif key.endswith('inform'):
                for act in key_acts:
                    acts.append([act[0].lower(), normalize(act[1])])
    return acts


def fix_delex(curr_dialog_acts, act_idx, text):
    """Given system dialogue acts fix automatic delexicalization."""
    if not act_idx in curr_dialog_acts:
        return text

    turn = curr_dialog_acts[act_idx]

    if isinstance(turn, dict):  # it's annotated:
        for key in turn:
            if 'Attraction' in key:
                if 'restaurant_' in text:
                    text = text.replace("restaurant", "attraction")
                if 'hotel_' in text:
                    text = text.replace("hotel", "attraction")
            if 'Hotel' in key:
                if 'attraction_' in text:
                    text = text.replace("attraction", "hotel")
                if 'restaurant_' in text:
                    text = text.replace("restaurant", "hotel")
            if 'Restaurant' in key:
                if 'attraction_' in text:
                    text = text.replace("attraction", "restaurant")
                if 'hotel_' in text:
                    text = text.replace("hotel", "restaurant")

    return text


def create_data(data_dir):
    data = json.load(open(f'{data_dir}/data.json', 'r'))
    dialog_acts = json.load(open(f'{data_dir}/dialogue_acts.json', 'r'))

    delex_data = {}

    for dialog_id in data:
        dialog = data[dialog_id]
        curr_dialog_acts = dialog_acts[dialog_id.strip('.json')]
        goals = [key for key in dialog['goal'].keys() if key in DOMAINS and dialog['goal'][key]]

        last_goal, act_idx = '', 1
        for idx, turn in enumerate(dialog['log']):
            dialog['log'][idx]['text'] = normalize(turn['text'])

            if idx % 2 == 1:  # system's turn
                cur_goal = get_goal(idx, dialog['log'], goals, last_goal)
                last_goal = cur_goal

                dialog['log'][idx - 1]['domain'] = cur_goal  # human's domain
                dialog['log'][idx]['dialogue_acts'] = get_dialog_act(curr_dialog_acts, str(act_idx))
                act_idx += 1

            dialog['log'][idx]['text'] = fix_delex(curr_dialog_acts, str(act_idx), dialog['log'][idx]['text'])

        delex_data[dialog_id] = dialog
    return delex_data


def analyze_dialogue(dialog, max_length):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    if len(dialog['log']) % 2 == 1:
        print('Odd number of turns. Wrong dialogue.')
        return None

    clean_dialog = {}
    clean_dialog['goal'] = dialog['goal']  # for now we just copy the goal
    usr_turns, sys_turns = [], []

    for idx in range(len(dialog['log'])):
        text = dialog['log'][idx]['text']
        if len(text.split()) > max_length or not is_ascii(text):
            return None  # sequence corrupted. discard

        if idx % 2 == 0:  # usr turn
            usr_turns.append(dialog['log'][idx])
        else:  # sys turn
            belief_summary, belief_value_summary = get_summary_belief_state(dialog['log'][idx]['metadata'])

            dialog['log'][idx]['belief_summary'] = str(belief_summary)
            dialog['log'][idx]['belief_value_summary'] = belief_value_summary
            sys_turns.append(dialog['log'][idx])

    clean_dialog['usr_log'] = usr_turns
    clean_dialog['sys_log'] = sys_turns

    return clean_dialog


def get_dialog(dialog, max_length=50):
    """Extract a dialogue from the file"""
    dialog = analyze_dialogue(dialog, max_length)
    if dialog is None:
        return None

    dialogs = []
    for idx in range(len(dialog['usr_log'])):
        dialogs.append(
            {
                'usr': dialog['usr_log'][idx]['text'],
                'sys': dialog['sys_log'][idx]['text'],
                'sys_a': dialog['sys_log'][idx]['dialogue_acts'],
                'domain': dialog['usr_log'][idx]['domain'],
                'bvs': dialog['sys_log'][idx]['belief_value_summary'],
            }
        )

    return dialogs


def partition_data(data, infold, outfold):
    """Partition the data into train, valid, and test sets
    based on the list of val and test specified in the dataset.
    """
    if if_exist(
        outfold, ['trainListFile.json', 'val_dialogs.json', 'test_dialogs.json', 'train_dialogs.json', 'ontology.json']
    ):
        print(f'Data is already processed and stored at {outfold}')
        return
    os.makedirs(outfold, exist_ok=True)
    shutil.copyfile(f'{infold}/ontology.json', f'{outfold}/ontology.json')

    with open(f'{infold}/testListFile.json', 'r') as fin:
        test_files = [line.strip() for line in fin.readlines()]

    with open(f'{infold}/valListFile.json', 'r') as fin:
        val_files = [line.strip() for line in fin.readlines()]

    train_list_files = open(f'{outfold}/trainListFile.json', 'w')

    train_dialogs, val_dialogs, test_dialogs = [], [], []
    count_train, count_val, count_test = 0, 0, 0

    for dialog_id in data:
        dialog = data[dialog_id]
        domains = [key for key in dialog['goal'].keys() if key in DOMAINS and dialog['goal'][key]]

        dial = get_dialog(dialog)
        if dial:
            dialogue = {}
            dialogue['dialog_idx'] = dialog_id
            dialogue['domains'] = list(set(domains))
            last_bs = []
            dialogue['dialog'] = []

            for idx, turn in enumerate(dial):
                turn_dl = {
                    'sys_transcript': dial[idx - 1]['sys'] if idx > 0 else "",
                    'turn_idx': idx,
                    'transcript': turn['usr'],
                    'sys_acts': dial[idx - 1]['sys_a'] if idx > 0 else [],
                    'domain': turn['domain'],
                }
                turn_dl['belief_state'] = [{"slots": [s], "act": "inform"} for s in turn['bvs']]
                turn_dl['turn_label'] = [bs["slots"][0] for bs in turn_dl['belief_state'] if bs not in last_bs]
                last_bs = turn_dl['belief_state']
                dialogue['dialog'].append(turn_dl)

            if dialog_id in test_files:
                test_dialogs.append(dialogue)
                count_test += 1
            elif dialog_id in val_files:
                val_dialogs.append(dialogue)
                count_val += 1
            else:
                train_list_files.write(dialog_id + '\n')
                train_dialogs.append(dialogue)
                count_train += 1

    print(f"Dialogs: {count_train} train, {count_val} val, {count_test} test.")

    # save all dialogues
    with open(f'{outfold}/val_dialogs.json', 'w') as fout:
        json.dump(val_dialogs, fout, indent=4)

    with open(f'{outfold}/test_dialogs.json', 'w') as fout:
        json.dump(test_dialogs, fout, indent=4)

    with open(f'{outfold}/train_dialogs.json', 'w') as fout:
        json.dump(train_dialogs, fout, indent=4)

    train_list_files.close()


def process_woz():
    delex_data = create_data(args.data_dir)
    partition_data(delex_data, args.data_dir, args.out_dir)


process_woz()
