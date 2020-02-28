7  #!/usr/bin/python

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
from os.path import exists, expanduser

from nemo.collections.nlp.data.datasets.datasets_utils import if_exist

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']

# List of the domains to process
DOMAINS = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[: sidx + 1] + ' ' + text[sidx + 1 :]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in REPLACEMENTS:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return data

    if not isinstance(turn, str):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")

    return data


def getDialogueAct(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    acts = []
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return acts

    if not isinstance(turn, str):
        for k in turn.keys():
            if k.split('-')[1].lower() == 'request':
                for a in turn[k]:
                    acts.append(a[0].lower())
            elif k.split('-')[1].lower() == 'inform':
                for a in turn[k]:
                    acts.append([a[0].lower(), normalize(a[1].lower())])
    return acts


def get_summary_bstate(bstate, get_domain=False):
    """Based on the mturk annotations we form multi-domain belief state"""
    summary_bstate = []
    summary_bvalue = []
    active_domain = []
    for domain in DOMAINS:
        domain_active = False

        booking = []
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if len(bstate[domain]['book']['booked']) != 0:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                    summary_bvalue.append(
                        [
                            "{}-book {}".format(domain, slot.strip().lower()),
                            normalize(bstate[domain]['book'][slot].strip().lower()),
                        ]
                    )
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                slot_enc[1] = 1
                summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()), "dontcare"])
            elif bstate[domain]['semi'][slot]:
                summary_bvalue.append(
                    [
                        "{}-{}".format(domain, slot.strip().lower()),
                        normalize(bstate[domain]['semi'][slot].strip().lower()),
                    ]
                )
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    assert len(summary_bstate) == 94
    if get_domain:
        return active_domain
    else:
        return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                return None
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                return None
            belief_summary, belief_value_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = str(belief_summary)
            d['log'][i]['belief_value_summary'] = belief_value_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    sys_a = [t['dialogue_acts'] for t in d_orig['sys_log']]
    bvs = [t['belief_value_summary'] for t in d_orig['sys_log']]
    domain = [t['domain'] for t in d_orig['usr_log']]
    for item in zip(usr, sys, sys_a, domain, bvs):
        dial.append({'usr': item[0], 'sys': item[1], 'sys_a': item[2], 'domain': item[3], 'bvs': item[4]})
    return dial


def getDomain(idx, log, domains, last_domain):
    if idx == 1:
        active_domains = get_summary_bstate(log[idx]["metadata"], True)
        crnt_doms = active_domains[0] if len(active_domains) != 0 else domains[0]
        return crnt_doms
    else:
        ds_diff = get_ds_diff(log[idx - 2]["metadata"], log[idx]["metadata"])
        if len(ds_diff.keys()) == 0:  # no clues from dialog states
            crnt_doms = last_domain
        else:
            crnt_doms = list(ds_diff.keys())
        return crnt_doms[0]


def get_ds_diff(prev_d, crnt_d):
    diff = {}
    if not prev_d or not crnt_d:
        return diff

    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        assert k1 == k2
        if v1 != v2:  # updated
            diff[k2] = v2
    return diff


def createData(source_data_dir):

    data = json.load(open(f'{source_data_dir}/data.json', 'r'))
    data2 = json.load(open(f'{source_data_dir}/dialogue_acts.json', 'r'))

    delex_data = {}

    for didx, dialogue_name in enumerate(data):

        dialogue = data[dialogue_name]

        domains = []
        for dom_k, dom_v in dialogue['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:  # check whether contains some goal entities
                domains.append(dom_k)

        idx_acts = 1
        last_domain, last_slot_fill = "", []
        for idx, turn in enumerate(dialogue['log']):
            origin_text = normalize(turn['text'])
            dialogue['log'][idx]['text'] = origin_text

            if idx % 2 == 1:  # if it's a system turn

                cur_domain = getDomain(idx, dialogue['log'], domains, last_domain)
                last_domain = [cur_domain]

                dialogue['log'][idx - 1]['domain'] = cur_domain
                dialogue['log'][idx]['dialogue_acts'] = getDialogueAct(dialogue_name, dialogue, data2, idx, idx_acts)
                idx_acts += 1

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)

        delex_data[dialogue_name] = dialogue

    return delex_data


def divideData(data, infold, outfold):
    """Given test and validation sets, divide
    the data for three different sets"""

    os.makedirs(outfold, exist_ok=True)
    shutil.copyfile(f'{infold}/ontology.json', f'{outfold}/ontology.json')

    testListFile = []
    fin = open(f'{infold}/testListFile.json', 'r')
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open(f'{infold}/valListFile.json', 'r')
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    test_dials = []
    val_dials = []
    train_dials = []

    count_train, count_val, count_test = 0, 0, 0

    for dialogue_name in data:
        dial_item = data[dialogue_name]
        domains = []
        for dom_k, dom_v in dial_item['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:  # check whether contains some goal entities
                domains.append(dom_k)

        dial = get_dial(data[dialogue_name])
        if dial:
            dialogue = {}
            dialogue['dialogue_idx'] = dialogue_name
            dialogue['domains'] = list(set(domains))
            last_bs = []
            dialogue['dialogue'] = []

            for turn_i, turn in enumerate(dial):
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_dialog = {}
                turn_dialog['system_transcript'] = dial[turn_i - 1]['sys'] if turn_i > 0 else ""
                turn_dialog['turn_idx'] = turn_i
                turn_dialog['belief_state'] = [{"slots": [s], "act": "inform"} for s in turn['bvs']]
                turn_dialog['turn_label'] = [bs["slots"][0] for bs in turn_dialog['belief_state'] if bs not in last_bs]
                turn_dialog['transcript'] = turn['usr']
                turn_dialog['system_acts'] = dial[turn_i - 1]['sys_a'] if turn_i > 0 else []
                turn_dialog['domain'] = turn['domain']
                last_bs = turn_dialog['belief_state']
                dialogue['dialogue'].append(turn_dialog)

            if dialogue_name in testListFile:
                test_dials.append(dialogue)
                count_test += 1
            elif dialogue_name in valListFile:
                val_dials.append(dialogue)
                count_val += 1
            else:
                train_dials.append(dialogue)
                count_train += 1

    # save all dialogues
    with open(f'{outfold}/dev_dials.json', 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open(f'{outfold}/test_dials.json', 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open(f'{outfold}/train_dials.json', 'w') as f:
        json.dump(train_dials, f, indent=4)

    print(f"Saving done. Generated dialogs: {count_train} train, {count_val} val, {count_test} test.")


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description='Process MultiWOZ dataset')
    parser.add_argument(
        "--source_data_dir", default='~/data/state_tracking/multiwoz2.1/MULTIWOZ2.1/MULTIWOZ2.1/', type=str
    )
    parser.add_argument("--target_data_dir", default='~/data/state_tracking/multiwoz2.1/', type=str)
    args = parser.parse_args()

    # Get absolute paths.
    abs_source_data_dir = expanduser(args.source_data_dir)
    abs_target_data_dir = expanduser(args.target_data_dir)

    if not exists(abs_source_data_dir):
        raise FileNotFoundError(f"{abs_source_data_dir} does not exist.")

    # Check if the files exist
    if if_exist(abs_target_data_dir, ['ontology.json', 'dev_dials.json', 'test_dials.json', 'train_dials.json']):
        print(f'Data is already processed and stored at {abs_target_data_dir}, skipping pre-processing.')
        exit(0)

    fin = open('mapping.pair', 'r')
    REPLACEMENTS = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        REPLACEMENTS.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

    print('Creating dialogues...')
    # Process MultiWOZ dataset
    delex_data = createData(abs_source_data_dir)
    # Divide data
    divideData(delex_data, abs_source_data_dir, abs_target_data_dir)
