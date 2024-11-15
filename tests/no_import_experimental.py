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
