"""
Implemented following the guide at https://www.promptingguide.ai/models/phi-2#phi-2-usage
"""

from nemo.collections.common.prompts.formatter import PromptFormatter


class Phi2QAPromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_qa"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Instruct: |message|\n",
            "slots": {
                "message": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"Output: |message|",
            "slots": {
                "message": str,
            },
        },
    }


class Phi2ChatPromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_chat"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Human: |message|",
            "slots": {
                "message": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"AI: |message|",
            "slots": {
                "message": str,
            },
        },
    }


class Phi2CodePromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_code"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"|message|\n",
            "slots": {
                "message": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": str,
            },
        },
    }
