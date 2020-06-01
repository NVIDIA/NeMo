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
https://github.com/thu-coai/ConvLab-2/blob/master/convlab2/util/multiwoz/multiwoz_slot_trans.py
"""

__all__ = ['REF_USR_DA', 'REF_SYS_DA']

REF_USR_DA = {
    'Attraction': {
        'area': 'Area',
        'type': 'Type',
        'name': 'Name',
        'entrance fee': 'Fee',
        'address': 'Addr',
        'postcode': 'Post',
        'phone': 'Phone',
    },
    'Hospital': {'department': 'Department', 'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'},
    'Hotel': {
        'type': 'Type',
        'parking': 'Parking',
        'pricerange': 'Price',
        'internet': 'Internet',
        'area': 'Area',
        'stars': 'Stars',
        'name': 'Name',
        'stay': 'Stay',
        'day': 'Day',
        'people': 'People',
        'address': 'Addr',
        'postcode': 'Post',
        'phone': 'Phone',
    },
    'Police': {'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone', 'name': 'Name'},
    'Restaurant': {
        'food': 'Food',
        'pricerange': 'Price',
        'area': 'Area',
        'name': 'Name',
        'time': 'Time',
        'day': 'Day',
        'people': 'People',
        'phone': 'Phone',
        'postcode': 'Post',
        'address': 'Addr',
    },
    'Taxi': {
        'leaveAt': 'Leave',
        'destination': 'Dest',
        'departure': 'Depart',
        'arriveBy': 'Arrive',
        'car type': 'Car',
        'phone': 'Phone',
    },
    'Train': {
        'destination': 'Dest',
        'day': 'Day',
        'arriveBy': 'Arrive',
        'departure': 'Depart',
        'leaveAt': 'Leave',
        'people': 'People',
        'duration': 'Time',
        'price': 'Ticket',
        'trainID': 'Id',
    },
}

REF_SYS_DA = {
    'Attraction': {
        'Addr': "address",
        'Area': "area",
        'Choice': "choice",
        'Fee': "entrance fee",
        'Name': "name",
        'Phone': "phone",
        'Post': "postcode",
        'Price': "pricerange",
        'Type': "type",
        'none': None,
        'Open': None,
    },
    'Hospital': {'Department': 'department', 'Addr': 'address', 'Post': 'postcode', 'Phone': 'phone', 'none': None},
    'Booking': {
        'Day': 'day',
        'Name': 'name',
        'People': 'people',
        'Ref': 'Ref',
        'Stay': 'stay',
        'Time': 'time',
        'none': None,
    },
    'Hotel': {
        'Addr': "address",
        'Area': "area",
        'Choice': "choice",
        'Internet': "internet",
        'Name': "name",
        'Parking': "parking",
        'Phone': "phone",
        'Post': "postcode",
        'Price': "pricerange",
        'Ref': "Ref",
        'Stars': "stars",
        'Type': "type",
        'none': None,
    },
    'Restaurant': {
        'Addr': "address",
        'Area': "area",
        'Choice': "choice",
        'Name': "name",
        'Food': "food",
        'Phone': "phone",
        'Post': "postcode",
        'Price': "pricerange",
        'Ref': "Ref",
        'none': None,
    },
    'Taxi': {
        'Arrive': "arriveBy",
        'Car': "taxi_types",
        'Depart': "departure",
        'Dest': "destination",
        'Leave': "leaveAt",
        'Phone': "taxi_phone",
        'none': None,
    },
    'Train': {
        'Arrive': "arriveBy",
        'Choice': "choice",
        'Day': "day",
        'Depart': "departure",
        'Dest': "destination",
        'Id': "trainID",
        'Leave': "leaveAt",
        'People': "people",
        'Ref': "Ref",
        'Time': "duration",
        'none': None,
        'Ticket': 'price',
    },
    'Police': {'Addr': "address", 'Post': "postcode", 'Phone': "phone"},
}
