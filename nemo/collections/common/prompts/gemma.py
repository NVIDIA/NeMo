"""
Implemented following the guide at https://www.promptingguide.ai/models/gemma#gemma-7b-prompt-format
"""

from nemo.collections.common.prompts.formatter import PromptFormatter


GEMMA_BOS = "<start_of_turn>"
GEMMA_END_OF_TURN = "<end_of_turn>"
GEMMA_NL = "\n\n"


class GemmaPromptFormatter(PromptFormatter):
    REGISTER_NAME = "gemma"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{GEMMA_BOS}user\n|MESSAGE|{GEMMA_END_OF_TURN}\n{GEMMA_BOS}model\n",
            "slots": {
                "|MESSAGE|": str,
            },
        },
        INFERENCE_ROLE: {
            # Note: that trailing NL is bothering me.
            "template": f"|MESSAGE|{GEMMA_END_OF_TURN}\n",
            "slots": {
                "|MESSAGE|": str,
            },
        },
    }
