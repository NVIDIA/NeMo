from nemo.lightning.peft import PEFT
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.utils import factory


@factory
def gpt_lora() -> PEFT:
    return LoRA()


__all__ = ["gpt_lora"]
