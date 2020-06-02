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
'''
import torch

from nemo import logging
from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.utils.decorators import add_port_docs
from nemo.core import VoidType, NeuralType, ChannelType, LengthsType, LogitsType, AxisType, AxisKind
from nemo.collections.nlp.utils.callback_utils import tensor2numpy

__all__ = ['UtteranceEncoderNM', 'TradeOutputNM']


class UtteranceEncoderNM(NonTrainableNM):
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
        return {
            # "history": NeuralType((axes=('ANY'), VoidType())
        }
        

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'src_ids': NeuralType(('B', 'T'), ChannelType()),
            'src_lens': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, data_desc, history):
        super().__init__()
        self.data_desc = data_desc
        self.history = history

    def forward(self):
        """
        Returns prediction of the TRADE Dialogue state tracking model in the human readable format
        Args:
            state (dict): state dictionary - see nemo.collections.nlp.data.datasets.multiwoz_dataset.default_state for the
                format
            data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
                and associated vocabulary
            encoder (nm): TRADE encoder
            decoder (nm): TRADE decoder
        Returns:
            dialogue_state (list): list of domain-slot_name-slot_value, for example ['hotel-area-east', 'hotel-stars-4']
        """
        history =self.history
        context = ' ; '.join([item[1].strip().lower() for item in history]).strip() + ' ;'
        context_ids = self.data_desc.vocab.tokens2ids(context.split())
        src_ids = torch.tensor(context_ids).unsqueeze(0).to(self._device)
        src_lens = torch.tensor(len(context_ids)).unsqueeze(0).to(self._device)
        return src_ids, src_lens 


class TradeOutputNM(NonTrainableNM):
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
        return {
            'point_outputs_pred': NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
            'gating_preds': NeuralType(('B', 'D', 'D'), LogitsType())
        }
        

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'trade_output': NeuralType( 
                elements_type=VoidType())
        }

    def __init__(self, data_desc):
        """
        data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
                and associated vocabulary
        """
        super().__init__()
        self.data_desc = data_desc


    def forward(self, gating_preds, point_outputs_pred):
        p_max = torch.argmax(point_outputs_pred, dim=-1)
        point_outputs_max_list = [tensor2numpy(p_max)]
        g_max = torch.argmax(gating_preds, axis=-1)
        gate_outputs_max_list = tensor2numpy(g_max)
        trade_output = self.get_human_readable_output(gate_outputs_max_list, point_outputs_max_list)[0]
        logging.info('TRADE output: %s', trade_output)
        return trade_output

    def get_human_readable_output(self, gating_preds, point_outputs_pred):
        '''
        Returns trade output in the human readable format
        Args:
            
            gatirng_preds (array): an array of gating predictions, TRADE model output
            point_outputs_pred (list of arrays): TRADE model output, contains predicted pointers
        Returns:
            output (list of strings): TRADE model output, each values represents domain-slot_name-slot_value, for example, ['hotel-pricerange-cheap', 'hotel-type-hotel']

        '''
        slots = self.data_desc.slots
        bi = 0
        predict_belief_bsz_ptr = []
        inverse_unpoint_slot = dict([(v, k) for k, v in data_desc.gating_dict.items()])

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

