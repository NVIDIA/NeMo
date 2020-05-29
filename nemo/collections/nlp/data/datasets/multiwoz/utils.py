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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/thu-coai/ConvLab-2
"""

import copy
import json
import os
import re
from difflib import SequenceMatcher

import torch

from nemo.collections.nlp.data.datasets.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from nemo.collections.nlp.data.datasets.multiwoz.state import default_state
from nemo.collections.nlp.utils.callback_utils import tensor2numpy
from nemo.utils import logging

__all__ = [
    'reformat_belief_state',
    'str_similar',
    'minDistance',
    'normalize_value',
    'special_match',
    'detect_requestable_slots',
    'init_session',
    'get_trade_prediction',
    'dst_update',
]


def init_session():
    """
    Restarts dialogue session
    Returns:
        empty system utterance, empty dialogue history and default empty dialogue state
    """
    return '', '', default_state()


def get_trade_prediction(state, data_desc, encoder, decoder):
    """
    Returns prediction of the TRADE Dialogue state tracking model in the human readable format
    Args:
        state (dict): state dictionary - see nemo.collections.nlp.data.datasets.multiwoz.default_state for the format
    Returns:
        dialogue_state (list): list of domain-slot_name-slot_value, for example ['hotel-area-east', 'hotel-stars-4']
    """
    context = ' ; '.join([item[1].strip().lower() for item in state['history']]).strip() + ' ;'
    context_ids = data_desc.vocab.tokens2ids(context.split())
    src_ids = torch.tensor(context_ids).unsqueeze(0).to(encoder._device)
    src_lens = torch.tensor(len(context_ids)).unsqueeze(0).to(encoder._device)
    outputs, hidden = encoder.forward(src_ids, src_lens)
    point_outputs, gate_outputs = decoder.forward(
        encoder_hidden=hidden, encoder_outputs=outputs, input_lens=src_lens, src_ids=src_ids
    )
    p_max = torch.argmax(point_outputs, dim=-1)
    point_outputs_max_list = [tensor2numpy(p_max)]
    g_max = torch.argmax(gate_outputs, axis=-1)
    gate_outputs_max_list = tensor2numpy(g_max)
    return get_human_readable_output(data_desc, gate_outputs_max_list, point_outputs_max_list)[0]


def dst_update(state, data_desc, user_uttr, encoder, decoder):
    """
    Updates dialogue state
    Args:
        state (dict): state dictionary - see nemo.collections.nlp.data.datasets.multiwoz.default_state for the format
        data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
            and associated vocabulary
        user_uttr (str): user utterance from the current turn
    Returns:
        state (dict): state dictionary - see nemo.collections.nlp.data.datasets.multiwoz.default_state for the format
    """
    prev_state = state
    dst_output = get_trade_prediction(state, data_desc, encoder, decoder)
    logging.info(f'TRADE DST output: {dst_output}')

    new_belief_state = reformat_belief_state(
        dst_output, copy.deepcopy(prev_state['belief_state']), data_desc.ontology_value_dict
    )
    state['belief_state'] = new_belief_state

    ## update request state based on the latest user utterance
    new_request_state = copy.deepcopy(state['request_state'])
    user_request_slot = detect_requestable_slots(user_uttr.lower(), data_desc.det_dict)
    for domain in user_request_slot:
        for key in user_request_slot[domain]:
            if domain not in new_request_state:
                new_request_state[domain] = {}
            if key not in new_request_state[domain]:
                new_request_state[domain][key] = user_request_slot[domain][key]
    state['request_state'] = new_request_state
    return state


def get_human_readable_output(
    data_desc, gating_preds, point_outputs_pred, gating_labels=None, point_outputs_labels=None
):
    '''
    To get trade output in the human readable format
    '''
    slots = data_desc.slots
    bi = 0
    predict_belief_bsz_ptr = []
    inverse_unpoint_slot = dict([(v, k) for k, v in data_desc.gating_dict.items()])

    for si, sg in enumerate(gating_preds[bi]):
        if sg == data_desc.gating_dict["none"]:
            continue
        elif sg == data_desc.gating_dict["ptr"]:
            pred = point_outputs_pred[0][0][si]

            pred = [data_desc.vocab.idx2word[x] for x in pred]

            st = []
            for e in pred:
                if e == 'EOS':
                    break
                else:
                    st.append(e)
            st = " ".join(st)
            if st == "none":
                continue
            else:
                predict_belief_bsz_ptr.append(slots[si] + "-" + str(st))
        else:
            predict_belief_bsz_ptr.append(slots[si] + "-" + inverse_unpoint_slot[sg])
    # predict_belief_bsz_ptr ['hotel-pricerange-cheap', 'hotel-type-hotel']
    output = [predict_belief_bsz_ptr]

    if gating_labels is None or point_outputs_labels is None:
        return output

    label_belief_bsz_ptr = []
    for si, sg in enumerate(gating_labels[bi]):
        if sg == data_desc.gating_dict["none"]:
            continue
        elif sg == data_desc.gating_dict["ptr"]:
            label = point_outputs_labels[0][0][si]
            label = [data_desc.vocab.idx2word[label[0]], data_desc.vocab.idx2word[label[1]]]

            st = []
            for e in label:
                if e == 'EOS':
                    break
                else:
                    st.append(e)
            st = " ".join(st)
            if st == "none":
                continue
            else:
                label_belief_bsz_ptr.append(slots[si] + "-" + str(st))
        else:
            label_belief_bsz_ptr.append(slots[si] + "-" + inverse_unpoint_slot[sg])
    # label_belief_bsz_ptr ['hotel-pricerange-cheap', 'hotel-type-hotel']
    output.append(label_belief_bsz_ptr)
    return output


def reformat_belief_state(raw_state, bs, value_dict):
    '''bs - belief_state
    '''
    for item in raw_state:
        item = item.lower()
        slist = item.split('-', 2)
        domain = slist[0].strip()
        slot = slist[1].strip()
        value = slist[2].strip()
        if domain not in bs:
            raise Exception('Error: domain <{}> not in belief state'.format(domain))
        dbs = bs[domain]
        assert 'semi' in dbs
        assert 'book' in dbs
        slot = REF_SYS_DA[domain.capitalize()].get(slot, slot)
        # reformat some slots
        if slot == 'arriveby':
            slot = 'arriveBy'
        elif slot == 'leaveat':
            slot = 'leaveAt'
        if slot in dbs['semi']:
            dbs['semi'][slot] = normalize_value(value_dict, domain, slot, value)
        elif slot in dbs['book']:
            dbs['book'][slot] = value
        elif slot.lower() in dbs['book']:
            dbs['book'][slot.lower()] = value
        else:
            logging.warning(
                'unknown slot name <{}> with value <{}> of domain <{}>\nitem: {}\n\n'.format(slot, value, domain, item)
            )
    return bs


def str_similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def minDistance(word1, word2):
    """The minimum edit distance between word 1 and 2."""
    if not word1:
        return len(word2 or '') or 0
    if not word2:
        return len(word1 or '') or 0
    size1 = len(word1)
    size2 = len(word2)
    tmp = list(range(size2 + 1))
    value = None
    for i in range(size1):
        tmp[0] = i + 1
        last = i
        for j in range(size2):
            if word1[i] == word2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
            last = tmp[j + 1]
            tmp[j + 1] = value
    return value


def normalize_value(value_set, domain, slot, value):
    """Normalized the value produced by NLU module to map it to the ontology value space.

    Args:
        value_set (dict):
            The value set of task ontology.
        domain (str):
            The domain of the slot-value pairs.
        slot (str):
            The slot of the value.
        value (str):
            The raw value detected by NLU module.
    Returns:
        value (str): The normalized value, which fits with the domain ontology.
    """
    slot = slot.lower()
    value = value.lower()
    value = ' '.join(value.split())
    try:
        assert domain in value_set
    except:
        raise Exception('domain <{}> not found in value set'.format(domain))
    if slot not in value_set[domain]:
        logging.warning('slot {} no in domain {}'.format(slot, domain))
        return value
        # raise Exception(
        #     'slot <{}> not found in db_values[{}]'.format(
        #         slot, domain))
    value_list = value_set[domain][slot]
    # exact match or containing match
    v = _match_or_contain(value, value_list)
    if v is not None:
        return v
    # some transfomations
    cand_values = _transform_value(value)
    for cv in cand_values:
        v = _match_or_contain(cv, value_list)
        if v is not None:
            logging.warning('slot value found via _match_or_contain')
            return v
    # special value matching
    v = special_match(domain, slot, value)
    if v is not None:
        logging.warning('slot value found via special_match')
        return v
    logging.warning('Failed: domain {} slot {} value {}, raw value returned.'.format(domain, slot, value))
    return value


def _transform_value(value):
    cand_list = []
    # a 's -> a's
    if " 's" in value:
        cand_list.append(value.replace(" 's", "'s"))
    # a - b -> a-b
    if " - " in value:
        cand_list.append(value.replace(" - ", "-"))
    # center <-> centre
    if value == 'center':
        cand_list.append('centre')
    elif value == 'centre':
        cand_list.append('center')
    # the + value
    if not value.startswith('the '):
        cand_list.append('the ' + value)
    return cand_list


def _match_or_contain(value, value_list):
    """match value by exact match or containing"""
    if value in value_list:
        return value
    for v in value_list:
        if v in value or value in v:
            return v
    # fuzzy match, when len(value) is large and distance(v1, v2) is small
    for v in value_list:
        d = minDistance(value, v)
        if (d <= 2 and len(value) >= 10) or (d <= 3 and len(value) >= 15):
            return v
    return None


def special_match(domain, slot, value):
    """special slot fuzzy matching"""
    matched_result = None
    if slot == 'arriveby' or slot == 'leaveat':
        matched_result = _match_time(value)
    elif slot == 'price' or slot == 'entrance fee':
        matched_result = _match_pound_price(value)
    elif slot == 'trainid':
        matched_result = _match_trainid(value)
    elif slot == 'duration':
        matched_result = _match_duration(value)
    return matched_result


def _match_time(value):
    """Return the time (leaveby, arriveat) in value, None if no time in value."""
    mat = re.search(r"(\d{1,2}:\d{1,2})", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    return None


def _match_trainid(value):
    """Return the trainID in value, None if no trainID."""
    mat = re.search(r"TR(\d{4})", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    return None


def _match_pound_price(value):
    """Return the price with pounds in value, None if no trainID."""
    mat = re.search(r"(\d{1,2},\d{1,2} pounds)", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    mat = re.search(r"(\d{1,2} pounds)", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    if "1 pound" in value.lower():
        return '1 pound'
    if 'free' in value:
        return 'free'
    return None


def _match_duration(value):
    """Return the durations (by minute) in value, None if no trainID."""
    mat = re.search(r"(\d{1,2} minutes)", value)
    if mat is not None and len(mat.groups()) > 0:
        return mat.groups()[0]
    return None


def detect_requestable_slots(observation, det_dic):
    """
    Finds slot values in the observation (user utterance)
    and adds the to rquested  slots list - needed  for Dialogue Policy
    """
    result = {}
    observation = observation.lower()
    _observation = ' {} '.format(observation)
    for value in det_dic.keys():
        _value = ' {} '.format(value.strip())
        if _value in _observation:
            key, domain = det_dic[value].split('-')
            if domain not in result:
                result[domain] = {}
            result[domain][key] = 0
    return result
