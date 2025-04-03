from functools import partial, lru_cache
from typing import Callable, Dict, Optional, Union, TypeVar, NamedTuple
from pathlib import Path
from copy import deepcopy
import os

import fsspec
from fsspec.spec import AbstractFileSystem
from huggingface_hub import HfFileSystem
import torch.nn as nn
import fiddle as fdl
from nemo_run import Config, Partial

from nemo.common.ckpt.resolver import get_checkpoint


StatefulT = TypeVar("StatefulT")

hf_fs = HfFileSystem()


class CheckpointType(NamedTuple):
    """Data structure to hold checkpoint path and filesystem information."""
    path: str
    checkpoint_type: str
    fs: AbstractFileSystem
    fs_path: str
    files: list[str]

    def __str__(self) -> str:
        return self.checkpoint_type


class CheckpointRegistry:
    def __init__(self):
        self.handlers: Dict[str, "CheckpointHandler"] = {}

    def register(self, name: str):
        """Class decorator to register a checkpoint handler"""

        def decorator(handler_cls):
            if not issubclass(handler_cls, CheckpointHandler):
                raise TypeError(
                    f"Handler must inherit from CheckpointHandler, got {handler_cls}"
                )
            self.handlers[name] = handler_cls()
            return handler_cls

        return decorator

    @lru_cache(maxsize=128)
    def detect_checkpoint_type(
        self,
        path: str,
        path_resolver: Optional[Union[Callable[[Path], Optional[Path]], str]] = None,
    ) -> CheckpointType:
        if path_resolver:
            resolved_path = get_checkpoint(path, path_resolver)
        else:
            resolved_path = path

        try:
            fs, fs_path = fsspec.core.url_to_fs(str(resolved_path))
            fs.ls(fs_path, detail=True)
            
            # Try to list files to check if path exists
            try:
                files = []
                for f in fs.ls(fs_path, detail=True):
                    if f["type"] == "directory":
                        files.extend(fs.ls(f["name"], detail=False))
                    else:
                        files.append(f["name"])

                # Convert to relative paths once
                rel_files = [os.path.relpath(f, fs_path) for f in files]

                for name, handler in self.handlers.items():
                    if handler.detect(fs, fs_path, rel_files):
                        return CheckpointType(
                            path=str(resolved_path),
                            checkpoint_type=name,
                            fs=fs,
                            fs_path=fs_path,
                            files=rel_files
                        )
            except FileNotFoundError:
                import pdb; pdb.set_trace()
                
                # If local path doesn't exist, try HuggingFace
                if ":://" not in resolved_path:
                    hf_path = f"hf://{resolved_path}"
                    return self.detect_checkpoint_type(hf_path, path_resolver)
                raise

        except FileNotFoundError:
            # If local path doesn't exist, try HuggingFace
            if ":://" not in resolved_path:
                hf_path = f"hf://{resolved_path}"
                return self.detect_checkpoint_type(hf_path, path_resolver)
            raise
        except Exception as e:
            raise ValueError(f"No checkpoint detector found for {path}: {str(e)}")

        raise ValueError(f"No checkpoint detector found for {path}")

    def load_context(
        self,
        path: str | Path,
        build: bool = False,
        path_resolver: Optional[Union[Callable[[Path], Optional[Path]], str]] = None,
        copy: bool = True
    ) -> Config:
        checkpoint_info = self.detect_checkpoint_type(str(path), path_resolver=path_resolver)
        checkpoint_type = checkpoint_info.checkpoint_type
        resolved_path = checkpoint_info.path

        cfg = self.handlers[checkpoint_type].load_context(str(resolved_path))

        if copy:
            cfg = deepcopy(cfg)

        if build:
            out = fdl.build(cfg)

            if isinstance(out, partial):
                return Partial(out)
            return out

        else:
            return cfg

    def init_model(
        self,
        path: str,
        path_resolver: Optional[Union[Callable[[Path], Optional[Path]], str]] = None,
        **kwargs
    ) -> nn.Module:
        checkpoint_info = self.detect_checkpoint_type(path, path_resolver=path_resolver)
        return self.handlers[checkpoint_info.checkpoint_type].init_model(path, **kwargs)


class CheckpointHandler:
    def detect(self, fs: AbstractFileSystem, path: str, files: list[str]) -> bool:
        """Implementation of detection logic"""
        raise NotImplementedError

    def load_context(self, path: str) -> Config:
        """Load the configuration for this checkpoint type"""
        raise NotImplementedError

    def load_model(self, path: str, **kwargs) -> nn.Module:
        """Load the model for this checkpoint type"""
        raise NotImplementedError


registry = CheckpointRegistry()
load_context = registry.load_context
detect_checkpoint_type = registry.detect_checkpoint_type
init_model = registry.load_model
