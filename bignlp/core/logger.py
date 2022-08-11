import os
import logging
from logging import config
from typing import Union

# provide a way to change level through BIGNLP_LOG_LEVEL environment variable:
# ...
LOG_VARNAME = "BIGNLP_LOG_LEVEL"
level_str = os.environ.get(LOG_VARNAME, "INFO").upper()
level: Union[int, str] = level_str if not level_str.isdigit() else int(level_str)

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"bignlp_basic": {"format": "%(name)s %(levelname)s (%(asctime)s) - %(message)s"}},
    "handlers": {
        "bignlp_out": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "bignlp_basic",
            "stream": "ext://sys.stdout",
        },
        "bignlp_err": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "bignlp_basic",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {"bignlp": {"handlers": ["bignlp_err", "bignlp_out"], "level": level}},
}


if level != "NOCONFIG":
    logging.config.dictConfig(CONFIG)


def get_logger() -> logging.Logger:
    return logging.getLogger("bignlp")


def exception(*args: str) -> None:
    get_logger().exception(*args)


def warning(*args: str) -> None:
    get_logger().warning(*args)


logger = get_logger()