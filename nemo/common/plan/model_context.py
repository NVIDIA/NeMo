from typing import TypeVar, Callable
import os
from pathlib import Path

import fiddle as fld
from torch import nn
import nemo_run as run
from nemo.common.ckpt.resolver import get_checkpoint
from nemo.common.ckpt.registry import load_context
from nemo.common.plan.plan import Plan

ModelT = TypeVar("ModelT", bound=nn.Module)


class ModelContext(Plan[ModelT]):
    """A Plan that represents a model source or target.
    
    This class handles loading context and provides a consistent interface
    for representing models in the import process.
    """
    def __init__(
        self, 
        path: str,         
        path_resolver: str | Callable[[str], Path] | None = None,
        model_class: type | None = None, 
    ):
        super().__init__()
        self.path = path
        self.path_resolver = path_resolver
        
        if model_class:
            self.model_class = model_class
        else:
            self.context: run.Config[ModelT] = load_context(self.path, path_resolver=path_resolver, build=False)
            # TODO: Improve this
            if hasattr(self.context, "model"):
                self.context = self.context.model
            _built = self()
            if hasattr(_built, "model_path"):
                self.model_class = _built.model_path()
            else:
                self.model_class = self.context.__fn_or_cls__

    def execute(self) -> ModelT:
        if hasattr(self, "context"):
            return fld.build(self.context)
        
        return load_context(self.path, build=True)

    def __repr__(self) -> str:
        if self.path_resolver:
            _path = get_checkpoint(self.path, self.path_resolver)
        else:
            _path = self.path
        class_path = self.model_class
        if hasattr(self.model_class, '__module__') and hasattr(self.model_class, '__name__'):
            class_path = f"{self.model_class.__module__}.{self.model_class.__name__}"
        return (
            f"ModelContext(\n"
            f"  (path): {_path}\n"
            f"  (class): {class_path}\n"
            f")"
        )
