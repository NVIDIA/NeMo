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

import random
from copy import deepcopy

from nemo import logging
from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.collections.nlp.data.datasets.multiwoz_dataset.dbquery import Database
from nemo.collections.nlp.data.datasets.multiwoz_dataset.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA
from nemo.collections.nlp.neural_types import *
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

__all__ = ['RuleBasedDPMMultiWOZ']

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


class RuleBasedDPMMultiWOZ(NonTrainableNM):
    """
     Rule-based bot. Implemented for Multiwoz dataset.
    """

    def __init__(self, data_dir: str, name: str = None):
        """
        Initializes the object
        Args:
            data_dir (str): path to data directory
            name: name of the modules (DEFAULT: none)
        """
        # Call base class constructor.
        NonTrainableNM.__init__(self, name=name)
        # Set init values of attributes.
        self.last_state = {}
        self.db = Database(data_dir)
        self.last_request_state = {}
        self.last_belief_state = {}
        self.recommend_flag = -1
        self.choice = ""

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            'belief_state': NeuralType(
                axes=[
                    AxisType(kind=AxisKind.Batch, is_list=True),
                    AxisType(
                        kind=DialogAxisKind.Domain, is_list=True
                    ),  # always 7 domains - but cannot set size with is_list!
                ],
                elements_type=MultiWOZBeliefState(),
            ),
            'request_state': NeuralType(
                axes=[AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Sequence, is_list=True)],
                elements_type=StringType(),
            ),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        system_acts (list): system actions in the format: [['Inform', 'Train', 'Day', 'wednesday'], []] [act, domain, slot, slot_value]
        belief_state (dict): dialogue state with slot-slot_values pairs for all domains
        """
        return {
            'belief_state': NeuralType(
                axes=[
                    AxisType(kind=AxisKind.Batch, is_list=True),
                    AxisType(
                        kind=DialogAxisKind.Domain, is_list=True
                    ),  # always 7 domains - but cannot set size with is_list!
                ],
                elements_type=MultiWOZBeliefState(),
            ),
            'system_acts': NeuralType(
                axes=[AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Sequence, is_list=True)],
                elements_type=StringType(),
            ),
        }

    def forward(self, belief_state, request_state):
        """
        Generated System Act and add it to the belief state
        Args:
            belief_state (dict): dialogue state with slot-slot_values pairs for all domains
            request_state (dict): requested slots dict
        Returns:
            belief_state (dict): updated belief state
            system_acts (list): DA(Dialog Act), in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """

        if self.recommend_flag != -1:
            self.recommend_flag += 1

        self.kb_result = {}
        DA = {}
        user_action = self.check_diff(belief_state, request_state)

        self.last_request_state = deepcopy(request_state)
        self.last_belief_state = deepcopy(belief_state)

        for user_act in user_action:
            domain, _ = user_act.split('-')

            # Respond to general greetings
            if domain == 'general':
                self._update_greeting(user_act, DA)

            # Book taxi for user
            elif domain == 'Taxi':
                self._book_taxi(belief_state, DA)

            elif domain == 'Booking':
                pass

            # User's talking about other domain
            elif domain != "Train":
                self._update_DA(user_act, user_action, belief_state, DA)

            # Info about train
            else:
                self._update_train(user_act, user_action, belief_state, DA)

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

        if DA == {}:
            DA = {'general-greet': [['none', 'none']]}
        system_acts = []
        for domain_intent, svs in DA.items():
            domain, intent = domain_intent.split('-')
            if not svs and domain == 'general':
                system_acts.append([intent, domain, 'none', 'none'])
            else:
                for slot, value in svs:
                    system_acts.append([intent, domain, slot, value])

        logging.debug("DPM output: %s", system_acts)
        logging.debug("Belief State after DPM: %s", belief_state)
        logging.debug("Request State after DPM: %s", request_state)
        return belief_state, system_acts

    def _update_greeting(self, user_act, DA):
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

    def _book_taxi(self, belief_state, DA):
        """ Book a taxi for user. """

        blank_info = []
        for info in ['departure', 'destination']:
            if belief_state['taxi']['semi'] == "":
                info = REF_USR_DA['Taxi'].get(info, info)
                blank_info.append(info)
        if belief_state['taxi']['semi']['leaveAt'] == "" and belief_state['taxi']['semi']['arriveBy'] == "":
            blank_info += ['Leave', 'Arrive']

        # Finish booking, tell user car type and phone number
        if len(blank_info) == 0:
            if 'Taxi-Inform' not in DA:
                DA['Taxi-Inform'] = []
            car = self.generate_car()
            phone_num = self.generate_phone_num(11)
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

    def _update_DA(self, user_act, user_action, belief_state, DA):
        """ Answer user's utterance about any domain other than taxi or train. """

        domain, intent_type = user_act.split('-')
        if domain.lower() not in belief_state.keys():
            return
        constraints = []
        for slot in belief_state[domain.lower()]['semi']:
            if belief_state[domain.lower()]['semi'][slot] != "":
                constraints.append([slot, belief_state[domain.lower()]['semi'][slot]])

        kb_result = self.db.query(domain.lower(), constraints)
        self.kb_result[domain] = deepcopy(kb_result)

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

        else:
            # There's no result matching user's constraint
            if len(kb_result) == 0:
                if (domain + "-NoOffer") not in DA:
                    DA[domain + "-NoOffer"] = []

                for slot in belief_state[domain.lower()]['semi']:
                    if belief_state[domain.lower()]['semi'][slot] != "" and belief_state[domain.lower()]['semi'][
                        slot
                    ] not in ["do nt care", "do n't care", "dontcare"]:
                        slot_name = REF_USR_DA[domain].get(slot, slot)
                        DA[domain + "-NoOffer"].append([slot_name, belief_state[domain.lower()]['semi'][slot]])

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
                for prop in belief_state[domain.lower()]['semi']:
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
                    for prop in belief_state[domain.lower()]['semi']:
                        if belief_state[domain.lower()]['semi'][prop] == "":
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

    def _update_train(self, user_act, user_action, belief_state, DA):
        constraints = []
        for time in ['leaveAt', 'arriveBy']:
            if belief_state['train']['semi'][time] != "":
                constraints.append([time, belief_state['train']['semi'][time]])

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
            if belief_state['train']['semi'][prop] == "":
                slot = REF_USR_DA['Train'].get(prop, prop)
                DA["Train-Request"].append([slot, '?'])
            else:
                constraints.append([prop, belief_state['train']['semi'][prop]])

        kb_result = self.db.query('train', constraints)
        self.kb_result['Train'] = deepcopy(kb_result)

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

    def check_diff(self, belief_state, request_state):
        user_action = {}
        if self.last_belief_state == {} and self.last_request_state == {}:
            for domain in belief_state:
                for slot in belief_state[domain]['book']:
                    if slot != 'booked' and belief_state[domain]['book'][slot] != '':
                        if (domain.capitalize() + "-Inform") not in user_action:
                            user_action[domain.capitalize() + "-Inform"] = []
                        if [
                            REF_USR_DA[domain.capitalize()].get(slot, slot),
                            belief_state[domain]['book'][slot],
                        ] not in user_action[domain.capitalize() + "-Inform"]:
                            user_action[domain.capitalize() + "-Inform"].append(
                                [REF_USR_DA[domain.capitalize()].get(slot, slot), belief_state[domain]['book'][slot],]
                            )
                for slot in belief_state[domain]['semi']:
                    if belief_state[domain]['semi'][slot] != "":
                        if (domain.capitalize() + "-Inform") not in user_action:
                            user_action[domain.capitalize() + "-Inform"] = []
                        if [
                            REF_USR_DA[domain.capitalize()].get(slot, slot),
                            belief_state[domain]['semi'][slot],
                        ] not in user_action[domain.capitalize() + "-Inform"]:
                            user_action[domain.capitalize() + "-Inform"].append(
                                [REF_USR_DA[domain.capitalize()].get(slot, slot), belief_state[domain]['semi'][slot],]
                            )
            for domain in request_state:
                for slot in request_state[domain]:
                    if (domain.capitalize() + "-Request") not in user_action:
                        user_action[domain.capitalize() + "-Request"] = []
                    if [REF_USR_DA[domain].get(slot, slot), '?'] not in user_action[domain.capitalize() + "-Request"]:
                        user_action[domain.capitalize() + "-Request"].append([REF_USR_DA[domain].get(slot, slot), '?'])

        else:
            for domain in belief_state:
                for slot in belief_state[domain]['book']:
                    if (
                        slot != 'booked'
                        and belief_state[domain]['book'][slot] != self.last_belief_state[domain]['book'][slot]
                    ):
                        if (domain.capitalize() + "-Inform") not in user_action:
                            user_action[domain.capitalize() + "-Inform"] = []
                        if [
                            REF_USR_DA[domain.capitalize()].get(slot, slot),
                            belief_state[domain]['book'][slot],
                        ] not in user_action[domain.capitalize() + "-Inform"]:
                            user_action[domain.capitalize() + "-Inform"].append(
                                [REF_USR_DA[domain.capitalize()].get(slot, slot), belief_state[domain]['book'][slot],]
                            )
                for slot in belief_state[domain]['semi']:
                    if (
                        belief_state[domain]['semi'][slot] != self.last_belief_state[domain]['semi'][slot]
                        and belief_state[domain]['semi'][slot] != ''
                    ):
                        if (domain.capitalize() + "-Inform") not in user_action:
                            user_action[domain.capitalize() + "-Inform"] = []
                        if [
                            REF_USR_DA[domain.capitalize()].get(slot, slot),
                            belief_state[domain]['semi'][slot],
                        ] not in user_action[domain.capitalize() + "-Inform"]:
                            user_action[domain.capitalize() + "-Inform"].append(
                                [REF_USR_DA[domain.capitalize()].get(slot, slot), belief_state[domain]['semi'][slot],]
                            )
            for domain in request_state:
                for slot in request_state[domain]:
                    if (domain not in self.last_request_state) or (slot not in self.last_request_state[domain]):
                        if (domain.capitalize() + "-Request") not in user_action:
                            user_action[domain.capitalize() + "-Request"] = []
                        if [REF_USR_DA[domain.capitalize()].get(slot, slot), '?'] not in user_action[
                            domain.capitalize() + "-Request"
                        ]:
                            user_action[domain.capitalize() + "-Request"].append(
                                [REF_USR_DA[domain.capitalize()].get(slot, slot), '?']
                            )
        return user_action

    def generate_phone_num(self, length):
        """ Generate phone number."""
        string = ""
        while len(string) < length:
            string += '0123456789'[random.randint(0, 999999) % 10]
        return string

    def generate_car(self):
        """ Generate a car for taxi booking. """
        car_types = ["toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen", "tesla"]
        p = random.randint(0, 999999) % len(car_types)
        return car_types[p]
