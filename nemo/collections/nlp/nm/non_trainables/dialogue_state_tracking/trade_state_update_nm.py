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
This file contains code artifacts adapted from the original implementation:
https://github.com/thu-coai/ConvLab-2/blob/master/convlab2/dst/trade/multiwoz/trade.py
https://github.com/thu-coai/ConvLab-2
'''
import copy
import re

import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.collections.nlp.data.datasets.multiwoz_dataset.multiwoz_slot_trans import REF_SYS_DA
from nemo.collections.nlp.neural_types import *
from nemo.collections.nlp.utils.callback_utils import tensor2numpy
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

__all__ = ['TradeStateUpdateNM']


class TradeStateUpdateNM(NonTrainableNM):
    """
     Takes the predictions of the TRADE Dialogue state tracking model, 
     generates human-readable model output and updates the dialogue
     state with the TRADE predcitions
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            'point_outputs_pred': NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
            'gating_preds': NeuralType(('B', 'D', 'D'), LogitsType()),
            'belief_state': NeuralType(
                axes=[
                    AxisType(kind=AxisKind.Batch, is_list=True),
                    AxisType(kind=DialogAxisKind.Domain, is_list=True),  # 7 domains
                ],
                elements_type=MultiWOZBeliefState(),
            ),
            'user_uttr': NeuralType(axes=[AxisType(kind=AxisKind.Batch, is_list=True)], elements_type=Utterance()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'belief_state': NeuralType(
                axes=[
                    AxisType(kind=AxisKind.Batch, is_list=True),
                    AxisType(kind=DialogAxisKind.Domain, is_list=True),  # 7 domains
                ],
                elements_type=MultiWOZBeliefState(),
            ),
            'request_state': NeuralType(
                axes=[AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Sequence, is_list=True)],
                elements_type=StringType(),
            ),
        }

    def __init__(self, data_desc):
        """
        Initializes the object
        Args:
            data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
                and associated vocabulary
        """
        super().__init__()
        self.data_desc = data_desc

    def forward(self, gating_preds, point_outputs_pred, belief_state, user_uttr):
        """
        Processes the TRADE model output and updates the dialogue (belief) state with the model's predictions
        Args:
            user_uttr (str): user utterance
            request_state (dict): contains requestsed slots-slot_value pairs for each domain
            belief_state (dict): dialgoue belief state, containt slot-slot value pair for all domains
            gating_preds (float): TRADE model gating predictions
            point_outputs_pred (float): TRADE model pointers predictions
        Returns:
            updated request_state (dict)
            updated belief_state (dict)
        """
        gate_outputs_max, point_outputs_max = self.get_trade_prediction(gating_preds, point_outputs_pred)
        trade_output = self.get_human_readable_output(gate_outputs_max, point_outputs_max)[0]
        logging.debug('TRADE output: %s', trade_output)

        new_belief_state = self.reformat_belief_state(
            trade_output, copy.deepcopy(belief_state), self.data_desc.ontology_value_dict
        )
        # update request state based on the latest user utterance
        # extract current user output
        new_request_state = self.detect_requestable_slots(user_uttr.lower(), self.data_desc.det_dict)
        logging.debug('Belief State after TRADE: %s', belief_state)
        logging.debug('Request State after TRADE: %s', new_request_state)
        return new_belief_state, new_request_state

    def get_trade_prediction(self, gating_preds, point_outputs_pred):
        """
        Takes argmax of the model's predictions
        Args:
            gating_preds (float): TRADE model gating predictions
            point_outputs_pred (float): TRADE model output, contains predicted pointers
        Returns:
            gate_outputs_max_list (array): list of the gating predicions
            point_outputs_max_list (list of arrays): each array contains the pointers predictions
        """
        p_max = torch.argmax(point_outputs_pred, dim=-1)
        point_outputs_max = [tensor2numpy(p_max)]
        g_max = torch.argmax(gating_preds, axis=-1)
        gate_outputs_max = tensor2numpy(g_max)
        return gate_outputs_max, point_outputs_max

    def get_human_readable_output(self, gating_preds, point_outputs_pred):
        """
        Returns trade output in the human readable format
        Args:
            gating_preds (array): an array of gating predictions, TRADE model output
            point_outputs_pred (list of arrays): TRADE model output, contains predicted pointers
        Returns:
            output (list of strings): TRADE model output, each values represents domain-slot_name-slot_value, 
                for example, ['hotel-pricerange-cheap', 'hotel-type-hotel']
        """
        slots = self.data_desc.slots
        bi = 0
        predict_belief_bsz_ptr = []
        inverse_unpoint_slot = dict([(v, k) for k, v in self.data_desc.gating_dict.items()])

        for si, sg in enumerate(gating_preds[bi]):
            if sg == self.data_desc.gating_dict["none"]:
                continue
            elif sg == self.data_desc.gating_dict["ptr"]:
                pred = point_outputs_pred[0][0][si]

                pred = [self.data_desc.vocab.idx2word[x] for x in pred]

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
        return output

    def reformat_belief_state(self, raw_state, bs, value_dict):
        '''
        Reformat TRADE model raw state into the default_state format
        Args:
            raw_state(list of strings): raw TRADE model output/state, each values represents domain-slot_name-slot_value,
                for example, ['hotel-pricerange-cheap', 'hotel-type-hotel']
            bs (dict): belief state - see nemo.collections.nlp.data.datasets.multiwoz.default_state for the format
            value_dict (dict): a dictionary of all slot values for MultiWOZ dataset
        Returns:
            bs (dict): reformatted belief state
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
                dbs['semi'][slot] = self.normalize_value(value_dict, domain, slot, value)
            elif slot in dbs['book']:
                dbs['book'][slot] = value
            elif slot.lower() in dbs['book']:
                dbs['book'][slot.lower()] = value
            else:
                logging.warning(
                    'unknown slot name <{}> with value <{}> of domain <{}>\nitem: {}\n\n'.format(
                        slot, value, domain, item
                    )
                )
        return bs

    def normalize_value(self, value_set, domain, slot, value):
        """Normalized the value produced by NLU module to map it to the ontology value space.
        Args:
            value_set (dict): The value set of task ontology.
            domain (str): The domain of the slot-value pairs.
            slot (str): The slot of the value.
            value (str): The raw value detected by NLU module.
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

        value_list = value_set[domain][slot]
        # exact match or containing match
        v = self._match_or_contain(value, value_list)
        if v is not None:
            return v
        # some transfomations
        cand_values = self._transform_value(value)
        for cv in cand_values:
            v = self._match_or_contain(cv, value_list)
            if v is not None:
                logging.warning('slot value found via _match_or_contain')
                return v
        # special value matching
        v = self.special_match(domain, slot, value)
        if v is not None:
            logging.warning('slot value found via special_match')
            return v
        logging.warning('Failed: domain {} slot {} value {}, raw value returned.'.format(domain, slot, value))
        return value

    def _transform_value(self, value):
        """makes clean up value transformations"""
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

    def minDistance(self, word1, word2):
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

    def _match_or_contain(self, value, value_list):
        """
        Matches value by exact match or containing
        Args:
            value (str): slot value
            value_list (list of str): list of possible slot_values
        Returns:
            matched value
        """
        if value in value_list:
            return value
        for v in value_list:
            if v in value or value in v:
                return v
        # fuzzy match, when len(value) is large and distance(v1, v2) is small
        for v in value_list:
            d = self.minDistance(value, v)
            if (d <= 2 and len(value) >= 10) or (d <= 3 and len(value) >= 15):
                return v
        return None

    def special_match(self, domain, slot, value):
        """special slot fuzzy matching"""
        matched_result = None
        if slot == 'arriveby' or slot == 'leaveat':
            matched_result = self._match_time(value)
        elif slot == 'price' or slot == 'entrance fee':
            matched_result = self._match_pound_price(value)
        elif slot == 'trainid':
            matched_result = self._match_trainid(value)
        elif slot == 'duration':
            matched_result = self._match_duration(value)
        return matched_result

    def _match_time(self, value):
        """Returns the time (leaveby, arriveat) in value, None if no time in value."""
        mat = re.search(r"(\d{1,2}:\d{1,2})", value)
        if mat is not None and len(mat.groups()) > 0:
            return mat.groups()[0]
        return None

    def _match_trainid(self, value):
        """Returns the trainID in value, None if no trainID."""
        mat = re.search(r"TR(\d{4})", value)
        if mat is not None and len(mat.groups()) > 0:
            return mat.groups()[0]
        return None

    def _match_pound_price(self, value):
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

    def _match_duration(self, value):
        """Return the durations (by minute) in value, None if no trainID."""
        mat = re.search(r"(\d{1,2} minutes)", value)
        if mat is not None and len(mat.groups()) > 0:
            return mat.groups()[0]
        return None

    def detect_requestable_slots(self, observation, det_dic):
        """
        Finds slot values in the observation (user utterance) and adds the to the requested slots list
        Args:
            observation (str): user utterance
            det_dic (dict):  a dictionary of slot_name + (slot_name_domain) value pairs from user dialogue acts
        Returns:
            result (dict): of the requested slots in a format: {domain: {slot_name}: 0}
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
