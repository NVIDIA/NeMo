# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


__all__ = [
    'deprecated',
]

import functools

import wrapt

from nemo.utils import logging

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


def deprecated(wrapped=None, version=None, explanation=None):
    """ Decorator class used for indicating that a function is deprecated and going to be removed.
    Tracks down which functions printed the warning and will print it only once per function.
    """

    if wrapped is None:
        return functools.partial(deprecated, version=version, explanation=explanation)

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        """
        Method prints the adequate warning (only once per function) when
        required and calls the function func, passing the original arguments,
        i.e. version and explanation.

        Args:
          version: Version in which the function will be removed (optional)
          explanation: Additional explanation (optional), e.g. use method ``blabla instead``.
        """

        # Check if we already warned about that function.
        if wrapped.__name__ not in _PRINTED_WARNING.keys():
            # Add to list so we won't print it again.
            _PRINTED_WARNING[wrapped.__name__] = True

            # Prepare the warning message.
            msg = "Function ``{}`` is deprecated.".format(wrapped.__name__)

            # Optionally, add version and alternative.
            if version is not None:
                msg = msg + " It is going to be removed in "
                msg = msg + "the {} version.".format(version)

            if explanation is not None:
                msg = msg + " " + explanation

            # Display the deprecated warning.
            logging.warning(msg)

        # Call the function.
        return wrapped(*args, **kwargs)

    return wrapper(wrapped)
