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

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.collections.nlp.neural_types import *
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

__all__ = ['SystemUtteranceHistoryUpdate']


class SystemUtteranceHistoryUpdate(NonTrainableNM):
    """
    Updates dialogue history with system utterance.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        user_uttr (str): user utterance
        sys_uttr (str): system utterace
        dialog_history (list): dialogue history, list of system and diaglogue utterances
        """
        return {
            'sys_uttr': NeuralType(
                axes=[AxisType(kind=AxisKind.Batch, is_list=True)], elements_type=SystemUtterance()
            ),
            'dialog_history': NeuralType(
                axes=(AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Time, is_list=True),),
                elements_type=AgentUtterance(),
            ),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        dialog_history (list): dialogue history, being a list of user and system utterances.
        """
        return {
            'dialog_history': NeuralType(
                axes=(AxisType(kind=AxisKind.Batch, is_list=True), AxisType(kind=AxisKind.Time, is_list=True),),
                elements_type=AgentUtterance(),
            ),
        }

    def __init__(self):
        """
        Initializes the object
        Args:
            data_desc (obj): data descriptor for MultiWOZ dataset, contains information about domains, slots, 
                    and associated vocabulary
        """
        super().__init__()

    def forward(self, sys_uttr, dialog_history):
        """
        Returns updated dialog history.
        Args:
            sys_uttr (str): system utterace
            dialog_history (list): dialogue history, list of user and system diaglogue utterances
        Returns:
            dialog_history (list): updated dialogue history, list of user and system diaglogue utterances
        """
        dialog_history.append(["sys", sys_uttr])
        logging.debug("Dialogue history: %s", dialog_history)

        return dialog_history
