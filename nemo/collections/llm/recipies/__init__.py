from nemo.collections.llm.recipies import (
    llama3_8b,
    llama3_8b_16k
)
from nemo.collections.llm.recipies.optim import adam
from nemo.collections.llm.recipies.log.default import default_log


__all__ = [
    "llama3_8b",
    "llama3_8b_16k",
    "adam",
    "default_log",
]
