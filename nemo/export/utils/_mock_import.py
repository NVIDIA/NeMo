import importlib
import logging
import sys
import types
from contextlib import contextmanager

LOGGER = logging.getLogger("NeMo")

"""
Utility to mock imports of unavailable modules.

Created for the purpose of using NeMo checkpoints produced with nvcr.io/nvidia/nemo:25.02.rc2
containers (or later) and used in the environments where Megatron-Core is not available. This
currently includes NIM containers.
"""


@contextmanager
def _mock_import(module: str):
    """
    Context manager to mock the import of a specified module if it is not available.

    Args:
        module (str): The name of the module to mock.

    Yields:
        Yields control back to the caller.
    """

    class DummyModule(types.ModuleType):

        def __getattr__(self, name):
            class Dummy:
                pass

            return Dummy

    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        LOGGER.warning(f"Module '{module}' is not available, mocking with a dummy module.")
        sys_modules_backup = sys.modules.copy()

        dummy_module = DummyModule("dummy")
        module_name, *submodules = module.split(".")
        sys.modules[module_name] = dummy_module
        modules_mocked = [module_name]
        for submodule in submodules:
            module_name += f".{submodule}"
            sys.modules[module_name] = dummy_module
            modules_mocked.append(module_name)

        yield

        # Restore the original sys.modules
        for module_name in modules_mocked:
            if module_name in sys_modules_backup:
                sys.modules[module_name] = sys_modules_backup[module_name]
            else:
                del sys.modules[module_name]
    else:
        yield
