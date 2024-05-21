"""
Implemented following the guide at https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct
"""

from nemo.collections.common.prompts.formatter import PromptFormatter


MISTRAL_BOS = "<s>"
MISTRAL_PROMPT_BEGIN = "[INST]"
MISTRAL_PROMPT_END = "[/INST]"
MISTRAL_END_OF_TURN = "</s>"
MISTRAL_NL = "\n\n"


class MistralPromptFormatter(PromptFormatter):
    REGISTER_NAME = "mistral"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "preamble": {
            "template": MISTRAL_BOS,
        },
        "user": {
            "template": f"{MISTRAL_PROMPT_BEGIN} |MESSAGE| {MISTRAL_PROMPT_END} ",
            "slots": {
                "|MESSAGE|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|MESSAGE|{MISTRAL_END_OF_TURN}",
            "slots": {
                "|MESSAGE|": str,
            },
        },
    }
