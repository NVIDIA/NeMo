import json
from typing import Any, Dict


def maybe_cast_to_int(x: Any):
    """Try to cast a value to int, if it fails, return the original value.

    Args:
        x (Any): The value to be casted.
    Returns:
        Any: The casted value or the original value if casting fails.
    """
    try:
        return int(x)
    except Exception:
        return x


def standardize_json_config(quant_cfg: Dict[str, Any]):
    """Standardize the quantization configuration JSON file to ensure
    compatibility with modelopt. Modifiy the input dictionary in place.

    Args:
        quant_cfg (Dict[str, Any]): The quantization config dictionary to standardize.
    """
    for key, value in quant_cfg.items():
        if key == "block_sizes":
            value = {maybe_cast_to_int(k): v for k, v in value.items()}
            quant_cfg[key] = value
        elif key in {"num_bits", "scale_bits"} and isinstance(value, list):
            value = tuple(value)
            quant_cfg[key] = value
        if isinstance(value, dict):
            standardize_json_config(value)
        elif isinstance(value, list):
            for x in value:
                if isinstance(x, dict):
                    standardize_json_config(x)


def load_quant_cfg(cfg_path: str):
    """Load quantization configuration from a JSON file and adjust for modelopt
    standard if necessary.

    Args:
        cfg_path (str): Path to the quantization config JSON file.

    Returns:
        dict: The loaded quantization configuration.
    """
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    standardize_json_config(cfg)
    return cfg
