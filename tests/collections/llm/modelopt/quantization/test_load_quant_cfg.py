import json
import tempfile

import pytest

from nemo.collections.llm.modelopt.quantization.quant_cfg_choices import get_quant_cfg_choices
from nemo.collections.llm.modelopt.quantization.utils import load_quant_cfg

QUANT_CFG_CHOICES = get_quant_cfg_choices()


@pytest.mark.parametrize("cfg_name", ["nvfp4", "fp8"])
def test_load_quant_cfg(cfg_name):
    """Test loading a quantization config from a JSON file."""

    quant_cfg_org = QUANT_CFG_CHOICES[cfg_name]

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(quant_cfg_org, temp_file)
        temp_file.flush()
        quant_cfg_loaded = load_quant_cfg(temp_file.name)
        assert quant_cfg_loaded == quant_cfg_org
