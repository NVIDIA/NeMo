# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import logging

from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

# Set up basic logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class ForbiddenImportChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = 'forbidden-import-checker'
    priority = -1
    msgs = {
        'C9001': (
            "Importing '%s' is forbidden.",
            'forbidden-import',
            'Used when a forbidden import is found.',
        )
    }

    # Forbidden module path
    FORBIDDEN_IMPORT = ['nemo.experimental', 'megatron.core.experimental', 'experimental']

    def __init__(self, linter=None):
        super().__init__(linter)
        logger.debug("ForbiddenImportChecker initialized")  # Initialization check

    def visit_import(self, node):
        """Check for forbidden imports using the 'import' statement."""

        for name, _ in node.names:
            if name in self.FORBIDDEN_IMPORTS:
                self.add_message('forbidden-import', node=node, args=(name,))

    def visit_importfrom(self, node):
        """Check for forbidden imports using the 'from X import Y' statement."""
        logger.debug(f"visit_importfrom called with: {node.modname}")

        for name, _ in node.names:
            if name in self.FORBIDDEN_IMPORTS:
                self.add_message('forbidden-import', node=node, args=(f"{node.modname}.{name}",))


def register(linter):
    """Register the custom checker."""
    logger.debug("Registering ForbiddenImportChecker")
    linter.register_checker(ForbiddenImportChecker(linter))
