# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import functools
import inspect
import threading
import types
import uuid
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
from cloudpickle import dump
from cloudpickle import load as pickle_load
from fiddle._src.experimental import serialization
from typing_extensions import Self

from nemo.lightning.io.artifact import DirOrStringArtifact, FileArtifact
from nemo.lightning.io.artifact.base import Artifact
from nemo.lightning.io.capture import IOProtocol

# Thread-local storage for artifacts directory
_thread_local = threading.local()


def extract_name(cls):
    return str(cls).split('.')[-1].rstrip('>').rstrip("'")


def _io_flatten_object(instance):
    try:
        serialization.dump_json(instance.__io__)
    except (serialization.UnserializableValueError, AttributeError) as e:
        if not hasattr(_thread_local, "local_artifacts_dir") or not hasattr(_thread_local, "output_path"):
            raise e

        local_artifact_path = Path(_thread_local.local_artifacts_dir) / f"{uuid.uuid4()}"
        output_path = _thread_local.output_path
        artifact_path = output_path / local_artifact_path
        with open(artifact_path, "wb") as f:
            dump(getattr(instance, "__io__", instance), f)
        return (str(local_artifact_path),), None

    return instance.__io__.__flatten__()


def _io_unflatten_object(values, metadata):
    assert hasattr(_thread_local, "output_dir")
    output_dir = _thread_local.output_dir

    if len(values) == 1:
        pickle_path = values[0]
        with open(Path(output_dir) / pickle_path, "rb") as f:
            return pickle_load(f)

    return fdl.Config.__unflatten__(values, metadata)


def _io_path_elements_fn(x):
    try:
        serialization.dump_json(x.__io__)
    except (serialization.UnserializableValueError, AttributeError):
        return (serialization.IdentityElement(),)

    return x.__io__.__path_elements__()


def _io_register_serialization(cls):
    serialization.register_node_traverser(
        cls,
        flatten_fn=_io_flatten_object,
        unflatten_fn=_io_unflatten_object,
        path_elements_fn=_io_path_elements_fn,
    )


def _io_init(self, **kwargs) -> fdl.Config[Self]:
    """
    Initializes the configuration object (`__io__`) with the captured arguments.

    Args:
        **kwargs: A dictionary of arguments that were captured during object initialization.

    Returns
    -------
        fdl.Config[Self]: The initialized configuration object.
    """
    try:
        return fdl.Config(type(self), **kwargs)
    except Exception as e:
        error_msg = (
            f"Error creating fdl.Config for {type(self).__name__}: {str(e)}\n"
            f"Arguments that caused the error: {kwargs}\n"
            f"This may be due to unsupported argument types or nested configurations."
        )
        raise RuntimeError(error_msg) from e


def _io_transform_args(self, init_fn, *args, **kwargs) -> Dict[str, Any]:
    """
    Transforms and captures the arguments passed to the `__init__` method, filtering out
    any arguments that are instances of `IOProtocol` or are dataclass fields with default
    factories.

    Args:
        init_fn (Callable): The original `__init__` method of the class.
        *args: Variable length argument list for the `__init__` method.
        **kwargs: Arbitrary keyword arguments for the `__init__` method.

    Returns
    -------
        Dict[str, Any]: A dictionary of the captured and transformed arguments.
    """
    sig = inspect.signature(init_fn)
    bound_args = sig.bind_partial(self, *args, **kwargs)
    config_kwargs = {k: v for k, v in bound_args.arguments.items() if k != "self"}

    to_del = []
    for key in config_kwargs:
        if isinstance(config_kwargs[key], IOProtocol):
            config_kwargs[key] = config_kwargs[key].__io__
        if is_dataclass(config_kwargs[key]):
            config_kwargs[key] = fdl_dc.convert_dataclasses_to_configs(config_kwargs[key], allow_post_init=True)
            # Check if the arg is a factory (dataclasses.field)
        if config_kwargs[key].__class__.__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
            to_del.append(key)

    for key in to_del:
        del config_kwargs[key]

    return config_kwargs


def _io_wrap_init(cls):
    """Wraps the __init__ method of a class to add IO functionality."""
    original_init = cls.__init__

    if getattr(cls, "__wrapped_init__", False):
        return cls

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        if hasattr(self, "io_transform_args"):
            cfg_kwargs = self.io_transform_args(original_init, *args, **kwargs)
        else:
            cfg_kwargs = _io_transform_args(self, original_init, *args, **kwargs)
        if hasattr(self, "io_init"):
            self.__io__ = self.io_init(**cfg_kwargs)
        else:
            self.__io__ = _io_init(self, **cfg_kwargs)

        original_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init
    cls.__wrapped_init__ = True
    return cls


def track_io(target, artifacts: Optional[list[Artifact]] = None):
    """
    Adds IO functionality to the target object or eligible classes in the target module
    by wrapping __init__ and registering serialization methods.

    Args:
        target (object or types.ModuleType): The target object or module to modify.

    Returns:
        object or types.ModuleType: The modified target with IO functionality added to eligible classes.

    Examples:
        >>> from nemo.collections.common import tokenizers
        >>> modified_tokenizers = track_io(tokenizers)
        >>> ModifiedWordTokenizer = track_io(tokenizers.WordTokenizer)
    """

    def _add_io_to_class(cls):
        if inspect.isclass(cls) and hasattr(cls, "__init__") and not hasattr(cls, "__io__"):
            if cls in [str, int, float, tuple, list, dict, bool, type(None)]:
                return cls

            cls = _io_wrap_init(cls)
            _io_register_serialization(cls)
            cls.__io_artifacts__ = artifacts or []
        return cls

    def _process_module(module):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and _is_defined_in_module_or_submodules(obj, module):
                setattr(module, name, _add_io_to_class(obj))
        return module

    def _is_defined_in_module_or_submodules(obj, module):
        return obj.__module__ == module.__name__ or obj.__module__.startswith(f"{module.__name__}.")

    if isinstance(target, types.ModuleType):
        return _process_module(target)
    elif inspect.isclass(target):
        return _add_io_to_class(target)
    else:
        raise TypeError("Target must be a module or a class")


# Registers all required classes with track_io functionality
try:
    # Track HF tokenizers
    from transformers import AutoTokenizer as HfAutoTokenizer
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

    for cls in [HfAutoTokenizer, LlamaTokenizer, LlamaTokenizerFast]:
        track_io(
            cls,
            artifacts=[
                FileArtifact(attr_name, required=False)
                for attr_name in ['vocab_file', 'merges_file', 'tokenizer_file', 'name_or_path']
            ],
        )

    from nemo.collections.common.tokenizers import AutoTokenizer

    track_io(
        AutoTokenizer,
        artifacts=[
            FileArtifact("vocab_file", required=False),
            FileArtifact("merges_file", required=False),
            DirOrStringArtifact("pretrained_model_name", required=False),
        ],
    )
except ImportError:
    pass


try:
    from nemo.collections.common.tokenizers import SentencePieceTokenizer

    track_io(SentencePieceTokenizer, artifacts=[FileArtifact("model_path")])
except ImportError:
    pass
