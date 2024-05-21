"""
Implemented following the guide at https://www.promptingguide.ai/models/phi-2#phi-2-usage
"""

from nemo.collections.common.prompts.formatter import PromptFormatter


class Phi2QAPromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_qa"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Instruct: |MESSAGE|\n",
            "slots": {
                "|MESSAGE|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"Output: |MESSAGE|",
            "slots": {
                "|MESSAGE|": str | None,
            },
        },
    }


class Phi2ChatPromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_chat"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Human: |MESSAGE|",
            "slots": {
                "|MESSAGE|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"AI: |MESSAGE|",
            "slots": {
                "|MESSAGE|": str | None,
            },
        },
    }


class Phi2CodePromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_code"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"|MESSAGE|\n",
            "slots": {
                "|MESSAGE|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|MESSAGE|",
            "slots": {
                "|MESSAGE|": str | None,
            },
        },
    }
