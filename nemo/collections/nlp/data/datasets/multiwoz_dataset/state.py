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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/thu-coai/ConvLab-2/blob/master/convlab2/util/multiwoz/state.py
"""

__all__ = ['default_state']


def default_state():
    state = dict(user_action=[], system_action=[], belief_state={}, request_state={}, terminated=False, history=[])
    state['belief_state'] = {
        "police": {"book": {"booked": []}, "semi": {}},
        "hotel": {
            "book": {"booked": [], "people": "", "day": "", "stay": ""},
            "semi": {"name": "", "area": "", "parking": "", "pricerange": "", "stars": "", "internet": "", "type": ""},
        },
        "attraction": {"book": {"booked": []}, "semi": {"type": "", "name": "", "area": ""}},
        "restaurant": {
            "book": {"booked": [], "people": "", "day": "", "time": ""},
            "semi": {"food": "", "pricerange": "", "name": "", "area": "",},
        },
        "hospital": {"book": {"booked": []}, "semi": {"department": ""}},
        "taxi": {"book": {"booked": []}, "semi": {"leaveAt": "", "destination": "", "departure": "", "arriveBy": ""}},
        "train": {
            "book": {"booked": [], "people": ""},
            "semi": {"leaveAt": "", "destination": "", "day": "", "arriveBy": "", "departure": ""},
        },
    }
    return state['system_action'], state['belief_state'], state['history']
