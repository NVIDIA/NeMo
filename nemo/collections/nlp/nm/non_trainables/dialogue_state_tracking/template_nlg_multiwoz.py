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
https://github.com/thu-coai/ConvLab-2/blob/master/convlab2/nlg/template/multiwoz/nlg.py
'''

import json
import os
import random

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.collections.nlp.neural_types import *
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

__all__ = ['TemplateNLGMultiWOZ']


# supported slot
slot2word = {
    'Fee': 'fee',
    'Addr': 'address',
    'Area': 'area',
    'Stars': 'stars',
    'Internet': 'Internet',
    'Department': 'department',
    'Choice': 'choice',
    'Ref': 'reference number',
    'Food': 'food',
    'Type': 'type',
    'Price': 'price range',
    'Stay': 'stay',
    'Phone': 'phone',
    'Post': 'postcode',
    'Day': 'day',
    'Name': 'name',
    'Car': 'car type',
    'Leave': 'leave',
    'Time': 'time',
    'Arrive': 'arrive',
    'Ticket': 'ticket',
    'Depart': 'departure',
    'People': 'people',
    'Dest': 'destination',
    'Parking': 'parking',
    'Open': 'open',
    'Id': 'Id',
    # 'TrainID': 'TrainID'
}


class TemplateNLGMultiWOZ(NonTrainableNM):
    """Generate a natural language utterance conditioned on the dialog act.
    """

    def __init__(self, mode="auto_manual", name=None):
        """
        Initializes the object
        Args:
            mode (str):
                - `auto`: templates extracted from data without manual modification, may have no match;
                - `manual`: templates with manual modification, sometimes verbose;
                - `auto_manual`: use auto templates first. When fails, use manual templates.
                both template are dict, *_template[dialog_act][slot] is a list of templates.
        """
        # Call base class constructor.
        NonTrainableNM.__init__(self, name=name)

        self.mode = mode
        template_dir = os.path.dirname(os.path.abspath(__file__))

        def read_json(filename):
            with open(filename, 'r') as f:
                return json.load(f)

        self.auto_system_template = read_json(os.path.join(template_dir, 'auto_system_template_nlg.json'))
        self.manual_system_template = read_json(os.path.join(template_dir, 'manual_system_template_nlg.json'))

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        system_acts (list): list of system actions action produced by dialog policy module
        """
        return {
            'system_acts': NeuralType(
                axes=[AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Sequence, is_list=True)],
                elements_type=StringType(),
            ),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        system_uttr (str): generated system's response
        """
        return {
            'sys_uttr': NeuralType(
                axes=[AxisType(kind=AxisKind.Batch, is_list=True)], elements_type=SystemUtterance()
            ),
        }

    def forward(self, system_acts):
        """
        Generates system response 
        Args:
            system_acts (list): system actions in the format: [['Inform', 'Train', 'Day', 'wednesday'], []] [act, domain, slot, slot_value]
        Returns:
            system_uttr (str): generated system utterance 
        """
        action = {}
        for intent, domain, slot, value in system_acts:
            k = '-'.join([domain, intent])
            action.setdefault(k, [])
            action[k].append([slot, value])
        dialog_acts = action
        mode = self.mode
        try:
            if mode == 'manual':
                system_uttr = self._manual_generate(dialog_acts, self.manual_system_template)

            elif mode == 'auto':
                system_uttr = self._auto_generate(dialog_acts, self.auto_system_template)

            elif mode == 'auto_manual':
                template1 = self.auto_system_template
                template2 = self.manual_system_template
                system_uttr = self._auto_generate(dialog_acts, template1)

                if system_uttr == 'None':
                    system_uttr = self._manual_generate(dialog_acts, template2)
            else:
                raise Exception("Invalid mode! available mode: auto, manual, auto_manual")
            # truncate a system utterance with multiple questions
            system_uttr = self.truncate_sys_response(system_uttr)
            logging.info("NLG output = System reply: %s", system_uttr)
            return system_uttr
        except Exception as e:
            logging.error('Error in processing: %s', dialog_acts)
            raise e

    def truncate_sys_response(self, sys_uttr):
        """
        Truncates system response when too many questions are asked by the system
        Args:
            sys_uttr (str): generated system response
        Returns:
            (str): updated system response
        """
        start_idx = 0
        utterances_with_period = []
        utterances_with_question_mark = []
        for idx, ch in enumerate(sys_uttr):
            if ch == '?':
                utterances_with_question_mark.append((sys_uttr[start_idx : idx + 1]).strip())
                start_idx = idx + 1
            elif ch == '.':
                utterances_with_period.append((sys_uttr[start_idx : idx + 1]).strip())
                start_idx = idx + 1

        if len(utterances_with_question_mark) > 0:
            utterances_with_question_mark = utterances_with_question_mark[:1]

        return ' '.join(utterances_with_period) + ' '.join(utterances_with_question_mark)

    def _postprocess(self, sen):
        sen_strip = sen.strip()
        sen = ''.join([val.capitalize() if i == 0 else val for i, val in enumerate(sen_strip)])
        if len(sen) > 0 and sen[-1] != '?' and sen[-1] != '.':
            sen += '.'
        sen += ' '
        return sen

    def _manual_generate(self, dialog_acts, template):
        sentences = ''
        for dialog_act, slot_value_pairs in dialog_acts.items():
            intent = dialog_act.split('-')
            if 'Select' == intent[1]:
                slot2values = {}
                for slot, value in slot_value_pairs:
                    slot2values.setdefault(slot, [])
                    slot2values[slot].append(value)
                for slot, values in slot2values.items():
                    if slot == 'none':
                        continue
                    sentence = 'Do you prefer ' + values[0]
                    for i, value in enumerate(values[1:]):
                        if i == (len(values) - 2):
                            sentence += ' or ' + value
                        else:
                            sentence += ' , ' + value
                    sentence += ' {} ? '.format(slot2word[slot])
                    sentences += sentence
            elif 'Request' == intent[1]:
                for slot, value in slot_value_pairs:
                    if dialog_act not in template or slot not in template[dialog_act]:
                        sentence = 'What is the {} of {} ? '.format(slot.lower(), dialog_act.split('-')[0].lower())
                        sentences += sentence
                    else:
                        sentence = random.choice(template[dialog_act][slot])
                        sentence = self._postprocess(sentence)
                        sentences += sentence
            elif 'general' == intent[0] and dialog_act in template:
                sentence = random.choice(template[dialog_act]['none'])
                sentence = self._postprocess(sentence)
                sentences += sentence
            else:
                for slot, value in slot_value_pairs:
                    if value in ["do nt care", "do n't care", "dontcare"]:
                        sentence = 'I don\'t care about the {} of the {}'.format(
                            slot.lower(), dialog_act.split('-')[0].lower()
                        )
                    elif dialog_act in template and slot in template[dialog_act]:
                        sentence = random.choice(template[dialog_act][slot])
                        sentence = sentence.replace('#{}-{}#'.format(dialog_act.upper(), slot.upper()), str(value))
                    else:
                        if slot in slot2word:
                            sentence = 'The {} is {} . '.format(slot2word[slot], str(value))
                        else:
                            sentence = ''
                    sentence = self._postprocess(sentence)
                    sentences += sentence
        return sentences.strip()

    def _auto_generate(self, dialog_acts, template):
        sentences = ''
        for dialog_act, slot_value_pairs in dialog_acts.items():
            key = ''
            for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
                key += s + ';'
            if dialog_act in template and key in template[dialog_act]:
                sentence = random.choice(template[dialog_act][key])
                if 'Request' in dialog_act or 'general' in dialog_act:
                    sentence = self._postprocess(sentence)
                    sentences += sentence
                else:
                    for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
                        if v != 'none':
                            sentence = sentence.replace('#{}-{}#'.format(dialog_act.upper(), s.upper()), v, 1)
                    sentence = self._postprocess(sentence)
                    sentences += sentence
            else:
                return 'None'
        return sentences.strip()
