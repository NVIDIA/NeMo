import json
import os
import random
from pprint import pprint

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.utils.decorators import add_port_docs

__all__ = ['TemplateNLGMultiWOZNM']


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


class TemplateNLGMultiWOZNM(NonTrainableNM):
    """Generate a natural language utterance conditioned on the dialog act.
        
        Args:
            action (list of list):
                The dialog action produced by dialog policy module, which is in dialog act format.
        Returns:
            utterance (str):
                A natural langauge utterance.
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

    def __init__(self, is_user=False, mode="auto_manual"):
        """
        Args:
            is_user:
                if dialog_act from user or system
            mode:
                - `auto`: templates extracted from data without manual modification, may have no match;
                - `manual`: templates with manual modification, sometimes verbose;
                - `auto_manual`: use auto templates first. When fails, use manual templates.
                both template are dict, *_template[dialog_act][slot] is a list of templates.
        """
        self.is_user = is_user
        self.mode = mode
        template_dir = os.path.dirname(os.path.abspath(__file__))

        def read_json(filename):
            with open(filename, 'r') as f:
                return json.load(f)

        self.auto_system_template = read_json(os.path.join(template_dir, 'auto_system_template_nlg.json'))
        self.manual_system_template = read_json(os.path.join(template_dir, 'manual_system_template_nlg.json'))

        # TODO remove user
        if is_user:
            self.manual_user_template = read_json(os.path.join(template_dir, 'manual_user_template_nlg.json'))
            self.auto_user_template = read_json(os.path.join(template_dir, 'auto_user_template_nlg.json'))

    def generate(self, dialog_acts):
        """NLG for Multiwoz dataset

        Args:
            dialog_acts in the format: [['Inform', 'Train', 'Day', 'wednesday'], []] [act, domain, slot, slot_value]
        Returns:
            generated sentence
        """
        action = {}
        for intent, domain, slot, value in dialog_acts:
            k = '-'.join([domain, intent])
            action.setdefault(k, [])
            action[k].append([slot, value])
        dialog_acts = action
        mode = self.mode
        try:
            is_user = self.is_user
            if mode == 'manual':
                if is_user:
                    template = self.manual_user_template
                else:
                    template = self.manual_system_template

                return self._manual_generate(dialog_acts, template)

            elif mode == 'auto':
                if is_user:
                    template = self.auto_user_template
                else:
                    template = self.auto_system_template

                return self._auto_generate(dialog_acts, template)

            elif mode == 'auto_manual':
                if is_user:
                    template1 = self.auto_user_template
                    template2 = self.manual_user_template
                else:
                    template1 = self.auto_system_template
                    template2 = self.manual_system_template

                res = self._auto_generate(dialog_acts, template1)
                if res == 'None':
                    res = self._manual_generate(dialog_acts, template2)
                return res

            else:
                raise Exception("Invalid mode! available mode: auto, manual, auto_manual")
        except Exception as e:
            print('Error in processing:')
            pprint(dialog_acts)
            raise e

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


def example():
    # dialog act
    dialog_acts = [['Inform', 'Train', 'Day', 'wednesday'], ['Inform', 'Train', 'Leave', '10:15']]
    print(dialog_acts)

    # system model for manual, auto, auto_manual
    nlg_sys_manual = TemplateNLG(is_user=False, mode='manual')
    nlg_sys_auto = TemplateNLG(is_user=False, mode='auto')
    nlg_sys_auto_manual = TemplateNLG(is_user=False, mode='auto_manual')

    # generate
    print('manual      : ', nlg_sys_manual.generate(dialog_acts))
    print('auto        : ', nlg_sys_auto.generate(dialog_acts))
    print('auto_manual : ', nlg_sys_auto_manual.generate(dialog_acts))


if __name__ == '__main__':
    example()
