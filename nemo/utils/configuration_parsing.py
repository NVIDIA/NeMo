# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# Copyright (C) IBM Corporation 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/IBM/pytorchpipe/blob/develop/ptp/configuration/config_parsing.py
"""

from nemo.utils.configuration_error import ConfigurationError


def get_value_list_from_dictionary(parameter: str, accepted_values=[]):
    """
    Parses parameter values retrieved from a given parameter dictionary using key.
    Optionally, checks is all values are accepted.

    Args:
        parameter: Value to be checked.
        accepted_values: List of accepted values (DEFAULT: [])

    Returns:
        List of parsed values
    """
    # Preprocess parameter value.
    if type(parameter) == str:
        if parameter == '':
            # Return empty list.
            return []
        else:
            # Process and split.
            values = parameter.replace(" ", "").split(",")
    else:
        values = parameter  # list
    if type(values) != list:
        raise ConfigurationError("'parameter' must be a list")

    # Test values one by one.
    if len(accepted_values) > 0:
        for value in values:
            if value not in accepted_values:
                raise ConfigurationError(
                    "One of the values in '{}' is invalid (current: '{}', accepted: {})".format(
                        key, value, accepted_values
                    )
                )

    # Return list.
    return values


def get_value_from_dictionary(parameter: str, accepted_values=[]):
    """
    Parses value of the parameter retrieved from a given parameter dictionary using key.
    Optionally, checks is the values is one of the accepted values.

    Args:
        parameter: Value to be checked.
        accepted_values: List of accepted values (DEFAULT: [])

    Returns:
        List of parsed values
    """
    if type(parameter) != str:
        raise ConfigurationError("'parameter' must be a string")
    # Preprocess parameter value.
    if parameter == '':
        return None

    # Test values one by one.
    if len(accepted_values) > 0:
        if parameter not in accepted_values:
            raise ConfigurationError(
                "One of the values in '{}' is invalid (current: '{}', accepted: {})".format(
                    key, value, accepted_values
                )
            )

    # Return value.
    return parameter
