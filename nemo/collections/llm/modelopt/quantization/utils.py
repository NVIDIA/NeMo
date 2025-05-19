import json
from pathlib import Path
from typing import Any, Dict, Union

from nemo.utils.cast_utils import maybe_cast_to_type


def standardize_json_config(quant_cfg: Dict[str, Any]):
    """Standardize the quantization configuration loaded from a JSON file to
    ensure compatibility with modelopt. Modifiy the input dictionary in place.

    Args:
        quant_cfg (Dict[str, Any]): The quantization config dictionary to standardize.
    """
    for key, value in quant_cfg.items():
        if key == "block_sizes":
            value = {maybe_cast_to_type(k, int): v for k, v in value.items()}
            quant_cfg[key] = value
        elif key in {"num_bits", "scale_bits"} and isinstance(value, list):
            quant_cfg[key] = tuple(value)
            continue  # No further processing needed
        if isinstance(value, dict):
            standardize_json_config(value)
        elif isinstance(value, list):
            for x in value:
                if isinstance(x, dict):
                    standardize_json_config(x)


def load_quant_cfg(cfg_path: Union[str, Path]) -> Dict[str, Any]:
    """Load quantization configuration from a JSON file and adjust for
    modelopt standards if necessary.

    Args:
        cfg_path (str): Path to the quantization config JSON file.

    Returns:
        dict: The loaded quantization configuration.
    """
    with open(cfg_path, "r") as f:
        quant_cfg = json.load(f)

    standardize_json_config(quant_cfg)
    return quant_cfg
