import pytest
from nemo.collections import vlm


def test_siglip_config_error():
    config = vlm.CLIPViTConfig(vision_model_type="siglip")
    assert config.add_class_token == False
    assert config.class_token_len == 0
    with pytest.raises(ValueError):
        config.configure_model()
