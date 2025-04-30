from typing import Optional

import numpy as np
import torch

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils.enum import PrettyStrEnum

# https://stackoverflow.com/a/77213071
MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

def hash_text(prev_hash: torch.Tensor, add_labels: torch.Tensor) -> torch.Tensor:
    return prev_hash * MULTIPLIER + INCREMENT + add_labels