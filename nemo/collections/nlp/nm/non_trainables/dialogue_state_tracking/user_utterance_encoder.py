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
from nemo.collections.nlp.neural_types import *
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

__all__ = ['UserUtteranceEncoder']


class UserUtteranceEncoder(NonTrainableNM):
    """
    Updates dialogue history with user utterance and encodes the history (system and user utterances) into 
    a flat list of tokens (per sample).
    Args:
        data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
            and associated vocabulary
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        user_uttr (str): user utterance
        dialog_history (list): dialogue history, list of system and diaglogue utterances
        """
        return {
            'user_uttr': NeuralType(axes=[AxisType(kind=AxisKind.Batch, is_list=True)], elements_type=UserUtterance()),
            'dialog_history': NeuralType(
                axes=(AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Time, is_list=True),),
                elements_type=AgentUtterance(),
            ),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        dialog_ids (int): token ids for dialogue history
        dialog_lens (int): length of the tokenized dialogue history
        dialog_history (list): dialogue history, being a list of user and system utterances.
        """
        return {
            # TODO: second dimension is not TIME - as it is the concatenation of list of words (ids) of  all utterances!
            'dialog_ids': NeuralType(('B', 'T'), elements_type=TokenIndex()),
            'dialog_lens': NeuralType(tuple('B'), elements_type=Length()),
            'dialog_history': NeuralType(
                axes=(AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Time, is_list=True),),
                elements_type=AgentUtterance(),
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

    def forward(self, user_uttr, dialog_history):
        """
        Returns dialogue utterances in the format accepted by the TRADE Dialogue state tracking model
        Args:
            dialog_history (list): dialogue history, list of system and diaglogue utterances
            user_uttr (str): user utterance
        Returns:
            dialog_ids (int): token ids for the whole dialogue history
            dialog_lens (int): length of the whole tokenized dialogue history
            dialog_history (list): updated dialogue history, list of system and diaglogue utterances
        """
        # TODO: why we update sys utterance, whereas we have only user utterance at that point?
        dialog_history.append(["user", user_uttr])
        logging.debug("Dialogue history: %s", dialog_history)

        context = ' ; '.join([item[1].strip().lower() for item in dialog_history]).strip() + ' ;'
        context_ids = self.data_desc.vocab.tokens2ids(context.split())
        dialog_ids = torch.tensor(context_ids).unsqueeze(0).to(self._device)
        dialog_lens = torch.tensor(len(context_ids)).unsqueeze(0).to(self._device)

        # logging.debug("!! dialog_ids: %s", dialog_ids)
        # logging.debug("!! dialog_lens: %s", dialog_lens)

        return dialog_ids, dialog_lens, dialog_history
