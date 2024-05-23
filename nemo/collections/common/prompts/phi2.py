"""
Implemented following the guide at https://www.promptingguide.ai/models/phi-2#phi-2-usage
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class Phi2QAPromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_qa"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Instruct: |message|\nOutput: ",
            "slots": {
                "message": Modality.Text,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


class Phi2ChatPromptFormatter(PromptFormatter):
    REGISTER_NAME = "phi2_chat"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Human: |message|\nAI: ",
            "slots": {
                "message": Modality.Text,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
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
                "message": Modality.Text,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
