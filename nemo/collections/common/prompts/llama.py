from typing import Type

from nemo.collections.common.prompts.formatter import PromptFormatter


class Llama2PromptFormatter(PromptFormatter):

    REGISTER_NAME = "llama2"

    INFERENCE_ROLE = "assistant"

    TEMPLATE = {
        "system": {
            "template": "<<SYS>>\n|SYSTEM|\n<</SYS>>\n",
            "slots": {
                "|SYSTEM|": str,
            },
        },
        "user": {
            "template": "[INST]\nUser:|USER|\n[/INST]\n\nAssistant:",
            "slots": {
                "|USER|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|TEXT|",
            "slots": {
                "|TEXT|": str,
            },
        },
    }
