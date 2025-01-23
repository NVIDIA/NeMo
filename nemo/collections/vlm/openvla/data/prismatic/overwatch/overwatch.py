"""
overwatch.py

Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""

import logging
import logging.config
import os
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union

# Overwatch Default Format String
RICH_FORMATTER, DATEFMT = "| >> %(message)s", "%m/%d [%H:%M:%S]"

# Set Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {**{0: "[*] "}, **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]}}

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that wraps logging & `accelerate.PartialState`."""
        from accelerate import PartialState

        # Note that PartialState is always safe to initialize regardless of `accelerate launch` or `torchrun`
        #   =>> However, might be worth actually figuring out if we need the `accelerate` dependency at all!
        self.logger, self.distributed_state = ContextAdapter(logging.getLogger(name), extra={}), PartialState()

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> only Log `INFO` on Main Process, `ERROR` on others!
        self.logger.setLevel(logging.INFO if self.distributed_state.is_main_process else logging.ERROR)

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self.distributed_state.on_main_process

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        return self.distributed_state.on_local_main_process

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        return self.distributed_state.main_process_first

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        return self.distributed_state.local_main_process_first

    def is_rank_zero(self) -> bool:
        return self.distributed_state.is_main_process

    def rank(self) -> int:
        return self.distributed_state.process_index

    def local_rank(self) -> int:
        return self.distributed_state.local_process_index

    def world_size(self) -> int:
        return self.distributed_state.num_processes


class PureOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that just wraps logging."""
        self.logger = ContextAdapter(logging.getLogger(name), extra={})

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> INFO
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def get_identity_ctx() -> Callable[..., Any]:
        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return identity

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self.get_identity_ctx()

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        return self.get_identity_ctx()

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    return DistributedOverwatch(name) if int(os.environ.get("WORLD_SIZE", -1)) != -1 else PureOverwatch(name)
