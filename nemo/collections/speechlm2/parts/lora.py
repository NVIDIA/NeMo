from omegaconf import DictConfig, ListConfig
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from nemo.utils import logging


def maybe_install_lora(model):
    """Add LoRA adapters to a model, using HuggingFace PEFT library."""
    if "lora" in model.cfg:
        assert hasattr(model, "cfg") and isinstance(model.cfg, DictConfig)
        assert hasattr(model, "llm") and isinstance(model.llm, PreTrainedModel)
        assert "prevent_freeze_params" in model.cfg and isinstance(model.cfg.prevent_freeze_params, (list, ListConfig))
        model.lora_config = LoraConfig(**model.cfg.lora)
        model.llm = get_peft_model(model.llm, model.lora_config)
        model.cfg.prevent_freeze_params.append(r"^.+\.lora_.+$")
        logging.info(f"LoRA adapter installed: {model.lora_config}")
