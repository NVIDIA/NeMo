# Copyright (C) tkornuta, NVIDIA AI Applications Team. All Rights Reserved.
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

__author__ = "Tomasz Kornuta"

from io import StringIO
import unittest
from unittest.mock import patch

from nemo.utils.decorators.deprecated import deprecated_function


@deprecated_function
def say_whee():
    print("Whee!")


class DeprecatedTestCase(unittest.TestCase):

    def test_deprecated_say_whee(self):
        """ Tests whether both std and err streams return the right values."""

        # Mock up both std and stderr streams.
        with patch('sys.stdout', new=StringIO()) as std_out:
            with patch('sys.stderr', new=StringIO()) as std_err:
                say_whee()

        # Check std output.
        self.assertEqual(std_out.getvalue().strip(),
                         'Whee!')

        # Check error output.
        self.assertEqual(std_err.getvalue().strip(),
                         'Function ``say_whee`` is depricated.')

# if __name__ == "__main__":
#    unittest.main(module=__name__, buffer=True, exit=False)
