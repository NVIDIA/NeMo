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


from io import StringIO
from unittest.mock import patch

from nemo.utils.decorators import deprecated
from tests.common_setup import NeMoUnitTest


class DeprecatedTestCase(NeMoUnitTest):
    def test_say_whee_deprecated(self):
        """ Tests whether both std and err streams return the right values
        when function is deprecated."""

        @deprecated()
        def say_whee():
            print("Whee!")

        # Mock up both std and stderr streams.
        with patch('sys.stdout', new=StringIO()) as std_out:
            with patch('sys.stderr', new=StringIO()) as std_err:
                say_whee()

        # Check std output.
        self.assertEqual(std_out.getvalue().strip(), "Whee!")

        # Check error output.
        self.assertEqual(std_err.getvalue().strip(), 'Function ``say_whee`` is deprecated.')

    def test_say_wow_twice_deprecated(self):
        """ Tests whether both std and err streams return the right values
        when a deprecated is called twice."""

        @deprecated()
        def say_wow():
            print("Woooow!")

        # Mock up both std and stderr streams - first call
        with patch('sys.stdout', new=StringIO()) as std_out:
            with patch('sys.stderr', new=StringIO()) as std_err:
                say_wow()

        # Check std output.
        self.assertEqual(std_out.getvalue().strip(), "Woooow!")

        # Check error output.
        self.assertEqual(std_err.getvalue().strip(), 'Function ``say_wow`` is deprecated.')

        # Second call.
        with patch('sys.stdout', new=StringIO()) as std_out:
            with patch('sys.stderr', new=StringIO()) as std_err:
                say_wow()

        # Check std output.
        self.assertEqual(std_out.getvalue().strip(), "Woooow!")

        # Check error output - should be empty.
        self.assertEqual(std_err.getvalue().strip(), '')

    def test_say_whoopie_deprecated_version(self):
        """ Tests whether both std and err streams return the right values
        when function is deprecated and version is provided. """

        @deprecated(version=0.1)
        def say_whoopie():
            print("Whoopie!")

        # Mock up both std and stderr streams.
        with patch('sys.stdout', new=StringIO()) as std_out:
            with patch('sys.stderr', new=StringIO()) as std_err:
                say_whoopie()

        # Check std output.
        self.assertEqual(std_out.getvalue().strip(), "Whoopie!")

        # Check error output.
        self.assertEqual(
            std_err.getvalue().strip(),
            "Function ``say_whoopie`` is deprecated. It is going to be removed in the 0.1 version.",
        )

    def test_say_kowabunga_deprecated_explanation(self):
        """ Tests whether both std and err streams return the right values
        when function is deprecated and additional explanation is provided. """

        @deprecated(explanation="Please use ``print_ihaa`` instead.")
        def say_kowabunga():
            print("Kowabunga!")

        # Mock up both std and stderr streams.
        with patch('sys.stdout', new=StringIO()) as std_out:
            with patch('sys.stderr', new=StringIO()) as std_err:
                say_kowabunga()

        # Check std output.
        self.assertEqual(std_out.getvalue().strip(), "Kowabunga!")

        # Check error output.
        self.assertEqual(
            std_err.getvalue().strip(), 'Function ``say_kowabunga`` is deprecated. Please use ``print_ihaa`` instead.'
        )
