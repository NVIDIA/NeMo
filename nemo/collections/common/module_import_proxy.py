"""
Lazy module loader.
Code adapted from TensorFlow.
"""

import importlib
import sys
import types
from typing import Any, Dict, Optional


class ModuleImportProxy(types.ModuleType):
    """Imports a module lazily (ie only if the module is actually used)."""

    def __init__(self, package: str, module_name: str, global_namespace: Dict[str, Any]):
        self._module_name = module_name
        self._package = package
        self._global_namespace = global_namespace
        self._loaded_module = None
        self._global_namespace[module_name] = self

    def _load_and_switch(self):
        """Load the moduel and switch this object's class."""
        if self._loaded_module is not None:
            return self._loaded_module
        module = importlib.import_module(self._package)
        module = getattr(module, self._module_name, module)
        self._loaded_module = module
        self._global_namespace[self._module_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load_and_switch()
        return getattr(module, item)

    def __dir__(self):
        module = self._load_and_switch()
        return dir(module)

    def __repr__(self) -> str:
        if self._loaded_module is None:
            return f'<ModuleProxy for "{self._module_name}" (unloaded)>'
        else:
            return repr(self._module_name)
