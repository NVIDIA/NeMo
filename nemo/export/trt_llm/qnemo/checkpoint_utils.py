import os
from pathlib import Path

from nemo.export.tarutils import TarPath

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "rank{}.safetensors"


def is_trtllm_checkpoint(path: str) -> bool:
    """Detect if a given path is TensorRT-LLM checkpoint based on config presence."""
    if os.path.isdir(path):
        path = Path(path)
    else:
        path = TarPath(path)
    config_path = path / CONFIG_NAME
    return config_path.exists()
