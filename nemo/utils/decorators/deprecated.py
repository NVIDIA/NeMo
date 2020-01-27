# Copyright (C) NVIDIA. All Rights Reserved.
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


import nemo


class deprecated(object):
    """ Decorator class used for indicating that a function is deprecated and going to be removed.
    Tracks down which functions printed the warning and will print it only once per function.
    """

    # Static variable - list of names of functions that we already printed
    # the warning for.
    warned_functions = []

    def __init__(self, version=None, explanation=None):
        """
        Constructor. Stores version and explanation into local variables.

        Args:
          version: Version in which the function will be removed (optional)
          explanation: Additional explanation (optional), e.g. use method ``blabla instead``.

        """
        self.version = version
        self.explanation = explanation

    def __call__(self, func):
        """
        Method prints the adequate warning (only once per function) when
        required and calls the function func, passing the original arguments.
        """

        def wrapper(*args, **kwargs):
            """
            Function prints the adequate warning and calls the function func,
            passing the original arguments.
            """
            # Check if we already warned about that function.
            if func.__name__ not in deprecated.warned_functions:
                # Add to list so we won't print it again.
                deprecated.warned_functions.append(func.__name__)

                # Prepare the warning message.
                msg = "Function ``{}`` is deprecated.".format(func.__name__)

                # Optionally, add version and alternative.
                if self.version is not None:
                    msg = msg + " It is going to be removed in "
                    msg = msg + "the {} version.".format(self.version)

                if self.explanation is not None:
                    msg = msg + " " + self.explanation

                # Display the deprecated warning.
                nemo.logging.warning(msg)

            # Call the function.
            return func(*args, **kwargs)

        return wrapper
