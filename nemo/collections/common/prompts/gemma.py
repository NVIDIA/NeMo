"""
Implemented following the guide at https://www.promptingguide.ai/models/gemma#gemma-7b-prompt-format
"""

from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

GEMMA_BOS = "<start_of_turn>"
GEMMA_END_OF_TURN = "<end_of_turn>"
GEMMA_NL = "\n\n"


class GemmaPromptFormatter(PromptFormatter):
    NAME = "gemma"
    OUTPUT_ROLE = "assistant"
    INSERT_BOS = True
    INSERT_EOS = True
    TEMPLATE = {
        "user": {
            "template": f"{GEMMA_BOS}user\n|message|{GEMMA_END_OF_TURN}\n{GEMMA_BOS}model\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            # Note: that trailing NL is bothering me.
            "template": f"|message|{GEMMA_END_OF_TURN}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn(Cut, GemmaPromptFormatter)
def gemma1(cut: Cut, prompt: GemmaPromptFormatter):
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut
    if cut.has_custom("context"):
        context = cut.context
    elif cut.has_custom("question"):
        context = cut.question
    else:
        context = cut.default_context

    turns = [{"role": "user", "slots": {"message": context}}]
    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "slots": {"message": answer}})

    return prompt.encode_dialog(turns)
