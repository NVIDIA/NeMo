# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from nemo.core.neural_types import AxisKindAbstract, ElementType, StringType

__all__ = [
    'DialogAxisKind',
    'Utterance',
    'UserUtterance',
    'SystemUtterance',
    'AgentUtterance',
    'SlotValue',
    'MultiWOZBeliefState',
]


class DialogAxisKind(AxisKindAbstract):
    """ Class containing definitions of axis kinds specialized for the dialog problem domain. """

    Domain = 7


class Utterance(StringType):
    """Element type representing an utterance (e.g. "Is there a train from Ely to Cambridge on Tuesday ?")."""


class UserUtterance(Utterance):
    """Element type representing a utterance expresesd by the user."""


class SystemUtterance(Utterance):
    """Element type representing an utterance produced by the system."""


class AgentUtterance(ElementType):
    """Element type representing utterance returned by an agent (user or system) participating in a dialog."""

    # def __str__(self):
    #    return "Utterance returned by an agent (user or system) participating in a dialog."

    # def fields(self):
    #    return ("agent", "utterance")


class SlotValue(ElementType):
    """Element type representing slot-value pair."""

    # def __str__(self):
    #    return "Slot-value pair"

    # def fields(self):
    #    return ("slot", "value")


class MultiWOZBeliefState(SlotValue):
    """Element type representing MultiWOZ belief state - one per domain."""

    # def fields(self):
    #    return ("book", "semi")
