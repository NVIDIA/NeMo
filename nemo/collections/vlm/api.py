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

"""Reuses API endpoints from nemo.collections.llm.api under the 'vlm' namespace."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from functools import wraps

import lightning.pytorch as pl
import nemo_run as run
import torch
from typing_extensions import Annotated

import nemo.lightning as nl
from nemo.collections.llm import api as llm_api


def create_wrapper(original_func: Callable, namespace: str, cli_name: Optional[str] = None) -> Callable:
    """Creates a wrapper function with the original signature and docstring, registered under a CLI namespace.

    This function wraps the provided `original_func` using `functools.wraps` to preserve its metadata
    (e.g., name, docstring, signature). It then applies a CLI decorator to register the wrapper under
    the specified `namespace` with the given `cli_name` (or the original function's name if not provided).

    Args:
        original_func (Callable): The original function to be wrapped.
        namespace (str): The CLI namespace under which the function will be registered.
        cli_name (Optional[str], optional): The name to use for the CLI command. Defaults to the original function's name.

    Returns:
        Callable: The decorated wrapper function, which can be used as a CLI command.
    """
    if cli_name is None:
        cli_name = original_func.__name__

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        return original_func(*args, **kwargs)

    return run.cli.entrypoint(namespace=namespace, name=cli_name)(wrapper)


train = create_wrapper(llm_api.train, "vlm")
pretrain = create_wrapper(llm_api.pretrain, "vlm")
finetune = create_wrapper(llm_api.finetune, "vlm")
validate = create_wrapper(llm_api.validate, "vlm")
prune = create_wrapper(llm_api.prune, "vlm")
distill = create_wrapper(llm_api.distill, "vlm")
ptq = create_wrapper(llm_api.ptq, "vlm")
deploy = create_wrapper(llm_api.deploy, "vlm")
evaluate = create_wrapper(llm_api.evaluate, "vlm")
import_ckpt = create_wrapper(llm_api.import_ckpt, "vlm", cli_name="import")
export_ckpt = create_wrapper(llm_api.export_ckpt, "vlm", cli_name="export")
generate = create_wrapper(llm_api.generate, "vlm")


__all__ = [
    "train",
    "pretrain",
    "finetune",
    "validate",
    "prune",
    "distill",
    "ptq",
    "deploy",
    "evaluate",
    "import_ckpt",
    "export_ckpt",
    "generate",
]
