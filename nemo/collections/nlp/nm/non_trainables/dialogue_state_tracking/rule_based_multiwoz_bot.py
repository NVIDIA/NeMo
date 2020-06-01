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
https://github.com/thu-coai/ConvLab-2/blob/master/convlab2/policy/rule/multiwoz/rule_based_multiwoz_bot.py

The code is based on:
Schatzmann, Jost, et al. "Agenda-based user simulation for bootstrapping a POMDP dialogue system."
Human Language Technologies 2007: The Conference of the North American Chapter of the Association for
Computational Linguistics;
Companion Volume, Short Papers. Association for Computational Linguistics, 2007.
'''
import json
import random
from copy import deepcopy

from nemo import logging
from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.collections.nlp.data.datasets.multiwoz_dataset.dbquery import Database
from nemo.collections.nlp.data.datasets.multiwoz_dataset.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from nemo.utils.decorators import add_port_docs

__all__ = ['RuleBasedMultiwozBotNM']

SELECTABLE_SLOTS = {
    'Attraction': ['area', 'entrance fee', 'name', 'type'],
    'Hospital': ['department'],
    'Hotel': ['area', 'internet', 'name', 'parking', 'pricerange', 'stars', 'type'],
    'Restaurant': ['area', 'name', 'food', 'pricerange'],
    'Taxi': [],
    'Train': [],
    'Police': [],
}

INFORMABLE_SLOTS = [
    "Fee",
    "Addr",
    "Area",
    "Stars",
    "Internet",
    "Department",
    "Choice",
    "Ref",
    "Food",
    "Type",
    "Price",
    "Stay",
    "Phone",
    "Post",
    "Day",
    "Name",
    "Car",
    "Leave",
    "Time",
    "Arrive",
    "Ticket",
    None,
    "Depart",
    "People",
    "Dest",
    "Parking",
    "Open",
    "Id",
]

REQUESTABLE_SLOTS = ['Food', 'Area', 'Fee', 'Price', 'Type', 'Department', 'Internet', 'Parking', 'Stars', 'Type']

# Information required to finish booking, according to different domain.
booking_info = {'Train': ['People'], 'Restaurant': ['Time', 'Day', 'People'], 'Hotel': ['Stay', 'Day', 'People']}


# Judge if user has confirmed a unique choice, according to different domain
token = {'Attraction': ['Name', 'Addr', ''], 'Hotel': ['Name',]}


class RuleBasedMultiwozBotNM(NonTrainableNM):
    """
     Rule-based bot. Implemented for Multiwoz dataset.
     Predict the next agent action given dialog state.
        Args:
            state (dict or list of list):
                when the policy takes dialogue state as input, the type is dict.
                else when the policy takes dialogue act as input, the type is list of list.
        Returns:
            action (list of list or str):
                when the policy outputs dialogue act, the type is list of list.
                else when the policy outputs utterance directly, the type is str.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {}

    recommend_flag = -1
    choice = ""

    def __init__(self, data_dir):
        self.last_state = {}
        self.db = Database(data_dir)

    def init_session(self):
        self.last_state = {}

    def predict(self, state):
        """
        Args:
            State, please refer to util/state.py
        Output:
            DA(Dialog Act), in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """
        '''
        {'user_action': 'I want to find a moderate hotel in the east and a cheap restaurant', 'system_action': [], 
        'belief_state': {'police': {'book': {'booked': []}, 'semi': {}}, 
        'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''}, 'semi': {'name': '', 'area': 'east', 'parking': '', 'pricerange': 'cheap', 'stars': '', 'internet': '', 'type': ''}}, 
        'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}}, 
        'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''}, 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}}, 
        'hospital': {'book': {'booked': []}, 'semi': {'department': ''}}, 'taxi': {'book': {'booked': []}, 'semi': {'leaveAt': '', 'destination': '', 'departure': '', 'arriveBy': ''}}, 
        'train': {'book': {'booked': [], 'people': ''}, 'semi': {'leaveAt': '', 'destination': '', 'day': '', 'arriveBy': '', 'departure': ''}}},
        'request_state': {}, 'terminated': False, 'history': [['sys', 'null'], ['user', 'I want to find a moderate hotel in the east and a cheap restaurant']]}
        '''

        if self.recommend_flag != -1:
            self.recommend_flag += 1

        self.kb_result = {}

        DA = {}

        if 'user_action' in state and (len(state['user_action']) > 0) and type(state['user_action']) is not str:
            user_action = {}
            for da in state['user_action']:
                i, d, s, v = da
                k = '-'.join((d, i))
                if k not in user_action:
                    user_action[k] = []
                user_action[k].append([s, v])
        else:
            user_action = check_diff(self.last_state, state)

        self.last_state = deepcopy(state)

        for user_act in user_action:
            domain, intent_type = user_act.split('-')

            # Respond to general greetings
            if domain == 'general':
                self._update_greeting(user_act, state, DA)

            # Book taxi for user
            elif domain == 'Taxi':
                self._book_taxi(user_act, state, DA)

            elif domain == 'Booking':
                self._update_booking(user_act, state, DA)

            # User's talking about other domain
            elif domain != "Train":
                self._update_DA(user_act, user_action, state, DA)

            # Info about train
            else:
                self._update_train(user_act, user_action, state, DA)

            # Judge if user want to book
            self._judge_booking(user_act, user_action, DA)

            if 'Booking-Book' in DA:
                if random.random() < 0.5:
                    DA['general-reqmore'] = []
                user_acts = []
                for user_act_ in DA:
                    if user_act_ != 'Booking-Book':
                        user_acts.append(user_act_)
                for user_act_ in user_acts:
                    del DA[user_act_]

        # print("Sys action: ", DA)

        if DA == {}:
            DA = {'general-greet': [['none', 'none']]}
        tuples = []
        for domain_intent, svs in DA.items():
            domain, intent = domain_intent.split('-')
            if not svs and domain == 'general':
                tuples.append([intent, domain, 'none', 'none'])
            else:
                for slot, value in svs:
                    tuples.append([intent, domain, slot, value])
        state['system_action'] = tuples
        return tuples

    def _update_greeting(self, user_act, state, DA):
        """ General request / inform. """
        _, intent_type = user_act.split('-')

        # Respond to goodbye
        if intent_type == 'bye':
            if 'general-bye' not in DA:
                DA['general-bye'] = []
            if random.random() < 0.3:
                if 'general-welcome' not in DA:
                    DA['general-welcome'] = []
        elif intent_type == 'thank':
            DA['general-welcome'] = []

    def _book_taxi(self, user_act, state, DA):
        """ Book a taxi for user. """

        blank_info = []
        for info in ['departure', 'destination']:
            if state['belief_state']['taxi']['semi'] == "":
                info = REF_USR_DA['Taxi'].get(info, info)
                blank_info.append(info)
        if (
            state['belief_state']['taxi']['semi']['leaveAt'] == ""
            and state['belief_state']['taxi']['semi']['arriveBy'] == ""
        ):
            blank_info += ['Leave', 'Arrive']

        # Finish booking, tell user car type and phone number
        if len(blank_info) == 0:
            if 'Taxi-Inform' not in DA:
                DA['Taxi-Inform'] = []
            car = generate_car()
            phone_num = generate_phone_num(11)
            DA['Taxi-Inform'].append(['Car', car])
            DA['Taxi-Inform'].append(['Phone', phone_num])
            return

        # Need essential info to finish booking
        request_num = random.randint(0, 999999) % len(blank_info) + 1
        if 'Taxi-Request' not in DA:
            DA['Taxi-Request'] = []
        for i in range(request_num):
            slot = REF_USR_DA.get(blank_info[i], blank_info[i])
            DA['Taxi-Request'].append([slot, '?'])

    def _update_booking(self, user_act, state, DA):
        pass

    def _update_DA(self, user_act, user_action, state, DA):
        """ Answer user's utterance about any domain other than taxi or train. """

        domain, intent_type = user_act.split('-')
        if domain.lower() not in state['belief_state'].keys():
            return
        constraints = []
        for slot in state['belief_state'][domain.lower()]['semi']:
            if state['belief_state'][domain.lower()]['semi'][slot] != "":
                constraints.append([slot, state['belief_state'][domain.lower()]['semi'][slot]])

        kb_result = self.db.query(domain.lower(), constraints)
        self.kb_result[domain] = deepcopy(kb_result)

        # print("\tConstraint: " + "{}".format(constraints))
        # print("\tCandidate Count: " + "{}".format(len(kb_result)))
        # if len(kb_result) > 0:
        #     print("Candidate: " + "{}".format(kb_result[0]))

        # print(state['user_action'])
        # Respond to user's request
        if intent_type == 'Request':
            if self.recommend_flag > 1:
                self.recommend_flag = -1
                self.choice = ""
            elif self.recommend_flag == 1:
                self.recommend_flag == 0
            if (domain + "-Inform") not in DA:
                DA[domain + "-Inform"] = []
            for slot in user_action[user_act]:
                if len(kb_result) > 0:
                    kb_slot_name = REF_SYS_DA[domain].get(slot[0], slot[0])
                    if kb_slot_name in kb_result[0]:
                        DA[domain + "-Inform"].append([slot[0], kb_result[0][kb_slot_name]])
                    else:
                        DA[domain + "-Inform"].append([slot[0], "unknown"])
                        # DA[domain + "-Inform"].append([slot_name, state['kb_results_dict'][0][slot[0].lower()]])

        else:
            # There's no result matching user's constraint
            # if len(state['kb_results_dict']) == 0:
            if len(kb_result) == 0:
                if (domain + "-NoOffer") not in DA:
                    DA[domain + "-NoOffer"] = []

                for slot in state['belief_state'][domain.lower()]['semi']:
                    if state['belief_state'][domain.lower()]['semi'][slot] != "" and state['belief_state'][
                        domain.lower()
                    ]['semi'][slot] not in ["do nt care", "do n't care", "dontcare"]:
                        slot_name = REF_USR_DA[domain].get(slot, slot)
                        DA[domain + "-NoOffer"].append(
                            [slot_name, state['belief_state'][domain.lower()]['semi'][slot]]
                        )

                p = random.random()

                # Ask user if he wants to change constraint
                if p < 0.3:
                    req_num = min(random.randint(0, 999999) % len(DA[domain + "-NoOffer"]) + 1, 3)
                    if domain + "-Request" not in DA:
                        DA[domain + "-Request"] = []
                    for i in range(req_num):
                        slot_name = REF_USR_DA[domain].get(
                            DA[domain + "-NoOffer"][i][0], DA[domain + "-NoOffer"][i][0]
                        )
                        DA[domain + "-Request"].append([slot_name, "?"])

            # There's exactly one result matching user's constraint
            # elif len(state['kb_results_dict']) == 1:
            elif len(kb_result) == 1:

                # Inform user about this result
                if (domain + "-Inform") not in DA:
                    DA[domain + "-Inform"] = []
                props = []
                for prop in state['belief_state'][domain.lower()]['semi']:
                    props.append(prop)
                property_num = len(props)
                if property_num > 0:
                    info_num = random.randint(0, 999999) % property_num + 1
                    random.shuffle(props)
                    for i in range(info_num):
                        slot_name = REF_USR_DA[domain].get(props[i], props[i])
                        # DA[domain + "-Inform"].append([slot_name, state['kb_results_dict'][0][props[i]]])
                        DA[domain + "-Inform"].append([slot_name, kb_result[0][props[i]]])

            # There are multiple resultes matching user's constraint
            else:
                p = random.random()

                # Recommend a choice from kb_list
                if p < 0.3:
                    if (domain + "-Inform") not in DA:
                        DA[domain + "-Inform"] = []
                    if (domain + "-Recommend") not in DA:
                        DA[domain + "-Recommend"] = []
                    DA[domain + "-Inform"].append(["Choice", str(len(kb_result))])
                    idx = random.randint(0, 999999) % len(kb_result)
                    # idx = 0
                    choice = kb_result[idx]
                    if domain in ["Hotel", "Attraction", "Police", "Restaurant"]:
                        DA[domain + "-Recommend"].append(['Name', choice['name']])
                    self.recommend_flag = 0
                    self.candidate = choice
                    props = []
                    for prop in choice:
                        props.append([prop, choice[prop]])
                    prop_num = min(random.randint(0, 999999) % 3, len(props))
                    # prop_num = min(2, len(props))
                    random.shuffle(props)
                    for i in range(prop_num):
                        slot = props[i][0]
                        string = REF_USR_DA[domain].get(slot, slot)
                        if string in INFORMABLE_SLOTS:
                            DA[domain + "-Recommend"].append([string, str(props[i][1])])

                # Ask user to choose a candidate.
                elif p < 0.5:
                    prop_values = []
                    props = []
                    # for prop in state['kb_results_dict'][0]:
                    for prop in kb_result[0]:
                        # for candidate in state['kb_results_dict']:
                        for candidate in kb_result:
                            if prop not in candidate:
                                continue
                            if candidate[prop] not in prop_values:
                                prop_values.append(candidate[prop])
                        if len(prop_values) > 1:
                            props.append([prop, prop_values])
                        prop_values = []
                    random.shuffle(props)
                    idx = 0
                    while idx < len(props):
                        if props[idx][0] not in SELECTABLE_SLOTS[domain]:
                            props.pop(idx)
                            idx -= 1
                        idx += 1
                    if domain + "-Select" not in DA:
                        DA[domain + "-Select"] = []
                    for i in range(min(len(props[0][1]), 5)):
                        prop_value = REF_USR_DA[domain].get(props[0][0], props[0][0])
                        DA[domain + "-Select"].append([prop_value, props[0][1][i]])

                # Ask user for more constraint
                else:
                    reqs = []
                    for prop in state['belief_state'][domain.lower()]['semi']:
                        if state['belief_state'][domain.lower()]['semi'][prop] == "":
                            prop_value = REF_USR_DA[domain].get(prop, prop)
                            reqs.append([prop_value, "?"])
                    i = 0
                    while i < len(reqs):
                        if reqs[i][0] not in REQUESTABLE_SLOTS:
                            reqs.pop(i)
                            i -= 1
                        i += 1
                    random.shuffle(reqs)
                    if len(reqs) == 0:
                        return
                    req_num = min(random.randint(0, 999999) % len(reqs) + 1, 2)
                    if (domain + "-Request") not in DA:
                        DA[domain + "-Request"] = []
                    for i in range(req_num):
                        req = reqs[i]
                        req[0] = REF_USR_DA[domain].get(req[0], req[0])
                        DA[domain + "-Request"].append(req)

    def _update_train(self, user_act, user_action, state, DA):
        constraints = []
        for time in ['leaveAt', 'arriveBy']:
            if state['belief_state']['train']['semi'][time] != "":
                constraints.append([time, state['belief_state']['train']['semi'][time]])

        if len(constraints) == 0:
            p = random.random()
            if 'Train-Request' not in DA:
                DA['Train-Request'] = []
            if p < 0.33:
                DA['Train-Request'].append(['Leave', '?'])
            elif p < 0.66:
                DA['Train-Request'].append(['Arrive', '?'])
            else:
                DA['Train-Request'].append(['Leave', '?'])
                DA['Train-Request'].append(['Arrive', '?'])

        if 'Train-Request' not in DA:
            DA['Train-Request'] = []
        for prop in ['day', 'destination', 'departure']:
            if state['belief_state']['train']['semi'][prop] == "":
                slot = REF_USR_DA['Train'].get(prop, prop)
                DA["Train-Request"].append([slot, '?'])
            else:
                constraints.append([prop, state['belief_state']['train']['semi'][prop]])

        kb_result = self.db.query('train', constraints)
        self.kb_result['Train'] = deepcopy(kb_result)

        # print(constraints)
        # print(len(kb_result))
        if user_act == 'Train-Request':
            del DA['Train-Request']
            if 'Train-Inform' not in DA:
                DA['Train-Inform'] = []
            for slot in user_action[user_act]:
                # Train_DA_MAP = {'Duration': "Time", 'Price': 'Ticket', 'TrainID': 'Id'}
                # slot[0] = Train_DA_MAP.get(slot[0], slot[0])
                slot_name = REF_SYS_DA['Train'].get(slot[0], slot[0])
                try:
                    DA['Train-Inform'].append([slot[0], kb_result[0][slot_name]])
                except Exception:
                    pass
            return
        if len(kb_result) == 0:
            if 'Train-NoOffer' not in DA:
                DA['Train-NoOffer'] = []
            for prop in constraints:
                DA['Train-NoOffer'].append([REF_USR_DA['Train'].get(prop[0], prop[0]), prop[1]])
            if 'Train-Request' in DA:
                del DA['Train-Request']
        elif len(kb_result) >= 1:
            if len(constraints) < 4:
                return
            if 'Train-Request' in DA:
                del DA['Train-Request']
            if 'Train-OfferBook' not in DA:
                DA['Train-OfferBook'] = []
            for prop in constraints:
                DA['Train-OfferBook'].append([REF_USR_DA['Train'].get(prop[0], prop[0]), prop[1]])

    def _judge_booking(self, user_act, user_action, DA):
        """ If user want to book, return a ref number. """
        if self.recommend_flag > 1:
            self.recommend_flag = -1
            self.choice = ""
        elif self.recommend_flag == 1:
            self.recommend_flag == 0
        domain, _ = user_act.split('-')
        for slot in user_action[user_act]:
            if domain in booking_info and slot[0] in booking_info[domain]:
                if 'Booking-Book' not in DA:
                    if domain in self.kb_result and len(self.kb_result[domain]) > 0:
                        if 'Ref' in self.kb_result[domain][0]:
                            DA['Booking-Book'] = [["Ref", self.kb_result[domain][0]['Ref']]]
                        else:
                            DA['Booking-Book'] = [["Ref", "N/A"]]
                        # TODO handle booking between multi turn


def check_diff(last_state, state):
    # print(state)
    user_action = {}
    if last_state == {}:
        for domain in state['belief_state']:
            for slot in state['belief_state'][domain]['book']:
                if slot != 'booked' and state['belief_state'][domain]['book'][slot] != '':
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [
                        REF_USR_DA[domain.capitalize()].get(slot, slot),
                        state['belief_state'][domain]['book'][slot],
                    ] not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append(
                            [
                                REF_USR_DA[domain.capitalize()].get(slot, slot),
                                state['belief_state'][domain]['book'][slot],
                            ]
                        )
            for slot in state['belief_state'][domain]['semi']:
                if state['belief_state'][domain]['semi'][slot] != "":
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [
                        REF_USR_DA[domain.capitalize()].get(slot, slot),
                        state['belief_state'][domain]['semi'][slot],
                    ] not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append(
                            [
                                REF_USR_DA[domain.capitalize()].get(slot, slot),
                                state['belief_state'][domain]['semi'][slot],
                            ]
                        )
        for domain in state['request_state']:
            for slot in state['request_state'][domain]:
                if (domain.capitalize() + "-Request") not in user_action:
                    user_action[domain.capitalize() + "-Request"] = []
                if [REF_USR_DA[domain].get(slot, slot), '?'] not in user_action[domain.capitalize() + "-Request"]:
                    user_action[domain.capitalize() + "-Request"].append([REF_USR_DA[domain].get(slot, slot), '?'])

    else:
        for domain in state['belief_state']:
            for slot in state['belief_state'][domain]['book']:
                if (
                    slot != 'booked'
                    and state['belief_state'][domain]['book'][slot] != last_state['belief_state'][domain]['book'][slot]
                ):
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [
                        REF_USR_DA[domain.capitalize()].get(slot, slot),
                        state['belief_state'][domain]['book'][slot],
                    ] not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append(
                            [
                                REF_USR_DA[domain.capitalize()].get(slot, slot),
                                state['belief_state'][domain]['book'][slot],
                            ]
                        )
            for slot in state['belief_state'][domain]['semi']:
                if (
                    state['belief_state'][domain]['semi'][slot] != last_state['belief_state'][domain]['semi'][slot]
                    and state['belief_state'][domain]['semi'][slot] != ''
                ):
                    if (domain.capitalize() + "-Inform") not in user_action:
                        user_action[domain.capitalize() + "-Inform"] = []
                    if [
                        REF_USR_DA[domain.capitalize()].get(slot, slot),
                        state['belief_state'][domain]['semi'][slot],
                    ] not in user_action[domain.capitalize() + "-Inform"]:
                        user_action[domain.capitalize() + "-Inform"].append(
                            [
                                REF_USR_DA[domain.capitalize()].get(slot, slot),
                                state['belief_state'][domain]['semi'][slot],
                            ]
                        )
        for domain in state['request_state']:
            for slot in state['request_state'][domain]:
                if (domain not in last_state['request_state']) or (slot not in last_state['request_state'][domain]):
                    if (domain.capitalize() + "-Request") not in user_action:
                        user_action[domain.capitalize() + "-Request"] = []
                    if [REF_USR_DA[domain.capitalize()].get(slot, slot), '?'] not in user_action[
                        domain.capitalize() + "-Request"
                    ]:
                        user_action[domain.capitalize() + "-Request"].append(
                            [REF_USR_DA[domain.capitalize()].get(slot, slot), '?']
                        )
    return user_action


def deduplicate(lst):
    i = 0
    while i < len(lst):
        if lst[i] in lst[0:i]:
            lst.pop(i)
            i -= 1
        i += 1
    return lst


def generate_phone_num(length):
    """ Generate phone number."""
    string = ""
    while len(string) < length:
        string += '0123456789'[random.randint(0, 999999) % 10]
    return string


def generate_car():
    """ Generate a car for taxi booking. """
    car_types = ["toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen", "tesla"]
    p = random.randint(0, 999999) % len(car_types)
    return car_types[p]


def fake_state():
    user_action = {'Hotel-Request': [['Name', '?']], 'Train-Inform': [['Day', 'don\'t care']]}
    from convlab2.util.multiwoz.state import default_state

    init_belief_state = default_state()['belief_state']
    kb_results = [None, None]
    kb_results[0] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'south'}
    kb_results[1] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'north'}
    state = {
        'user_action': user_action,
        'belief_state': init_belief_state,
        'kb_results_dict': kb_results,
        'hotel-request': [['phone']],
    }
    '''
    state = {'user_action': dict(),
             'belief_state: dict(),
             'kb_results_dict': kb_results
    }
    '''
    return state


def test_init_state():
    user_action = ['general-hello']
    current_slots = dict()
    kb_results = [None, None]
    kb_results[0] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'south'}
    kb_results[1] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'north'}
    state = {'user_action': user_action, 'current_slots': current_slots, 'kb_results_dict': []}
    return state


def test_run():
    policy = RuleBasedMultiwozBot()
    system_act = policy.predict(fake_state())
    print(json.dumps(system_act, indent=4))
