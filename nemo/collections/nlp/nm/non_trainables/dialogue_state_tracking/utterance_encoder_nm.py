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
https://github.com/thu-coai/ConvLab-2/
'''
import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core import ChannelType, LengthsType, NeuralType, VoidType
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

__all__ = ['UtteranceEncoderNM']


class UtteranceEncoderNM(NonTrainableNM):
    """
    Encodes dialogue history (system and user utterances) into a Multiwoz dataset format
    Args:
        data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
            and associated vocabulary
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        state (dict): dialogue state dictionary - see nemo.collections.nlp.data.datasets.multiwoz_dataset.state
            for the format
        user_uttr (str): user utterance
        sys_uttr (str): system utterace
        """
        return {
            "state": NeuralType(axes=tuple('ANY'), elements_type=VoidType()),
            "user_uttr": NeuralType(axes=tuple('ANY'), elements_type=VoidType()),
            "sys_uttr": NeuralType(axes=tuple('ANY'), elements_type=VoidType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        src_ids (int): token ids for dialogue history
        src_lens (int): length of the tokenized dialogue history
        """
        return {
            'src_ids': NeuralType(('B', 'T'), elements_type=ChannelType()),
            'src_lens': NeuralType(tuple('B'), elemenets_type=LengthsType()),
        }

    def __init__(self, data_desc):
        """
        Init
        Args:
            data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
                    and associated vocabulary
        """
        super().__init__()
        self.data_desc = data_desc

    def forward(self, state, user_uttr, sys_uttr):
        """
        Returns dialogue utterances in the format accepted by the TRADE Dialogue state tracking model
        Args:
            state (dict): state dictionary - see nemo.collections.nlp.data.datasets.multiwoz_dataset.state
                for the format
            user_uttr (str): user utterance
            sys_uttr (str): system utterace
        Returns:
            src_ids (int): token ids for dialogue history
            src_lens (int): length of the tokenized dialogue history
        """
        state["history"].append(["sys", sys_uttr])
        state["history"].append(["user", user_uttr])
        state["user_action"] = user_uttr
        logging.debug("Dialogue state: %s", state)

        context = ' ; '.join([item[1].strip().lower() for item in state['history']]).strip() + ' ;'
        context_ids = self.data_desc.vocab.tokens2ids(context.split())
        src_ids = torch.tensor(context_ids).unsqueeze(0).to(self._device)
        src_lens = torch.tensor(len(context_ids)).unsqueeze(0).to(self._device)
        return src_ids, src_lens
