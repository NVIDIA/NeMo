# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Callable, Optional

from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.core.config_store import ConfigStore
from hydra.types import TaskFunction
from omegaconf import DictConfig

from nemo.core.config import Config


def hydra_runner(
    config_path: Optional[str] = None, config_name: Optional[str] = None, schema: Optional[Any] = None
) -> Callable[[TaskFunction], Any]:
    """
    Decorator used for passing the Config paths to main function.
    Optionally registers a schema used for validation/providing default values.

    Args:
        config_path: Path to the directory where the config exists.
        config_name: Name of the config file.
        schema: Structured config  type representing the schema used for validation/providing default values.
    """
    if schema is not None:
        # Create config store.
        cs = ConfigStore.instance()
        # Register the configuration as a node under a given name.
        cs.store(name=config_name.replace(".yaml", ""), node=schema)

    def decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def wrapper(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            # Check it config was passed.
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args = get_args_parser()

                # Parse arguments in order to retrieve overrides
                parsed_args = args.parse_args()

                # Get overriding args in dot string format
                overrides = parsed_args.overrides  # type: list

                # Update overrides
                overrides.append("hydra.run.dir=.")
                overrides.append('hydra.job_logging.root.handlers=null')

                # Wrap a callable object with name `parse_args`
                # This is to mimic the ArgParser.parse_args() API.
                class _argparse_wrapper:
                    def __init__(self, arg_parser):
                        self.arg_parser = arg_parser
                        self._actions = arg_parser._actions

                    def parse_args(self, args=None, namespace=None):
                        return parsed_args

                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                _run_hydra(
                    args_parser=_argparse_wrapper(args),
                    task_function=task_function,
                    config_path=config_path,
                    config_name=config_name,
                    strict=None,
                )

        return wrapper

    return decorator


def set_config(config: Config) -> Callable[[TaskFunction], Any]:
    """
    Decorator used for passing the Structured Configs to main function.

    Args:
        config: config class derived from Config.
    """
    # Get class name. Not sure how important this is, but coudn't get name by accessing type().__name__.
    class_name = str(config)
    # Create config store.
    cs = ConfigStore.instance()
    # Register the configuration as a node under a given name.
    cs.store(name=class_name, node=config)

    def decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def wrapper(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            # Check it config was passed.
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args = get_args_parser()

                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                _run_hydra(
                    args_parser=args,
                    task_function=task_function,
                    config_path=None,
                    config_name=class_name,
                    strict=None,
                )

        return wrapper

    return decorator
