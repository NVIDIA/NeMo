"""
Implemented following the guide at https://www.promptingguide.ai/models/gemma#gemma-7b-prompt-format
"""

from collections import defaultdict

from lhotse import CutSet
from lhotse.cut import MixedCut

from nemo.collections.common.prompts import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec

GEMMA_BOS = "<start_of_turn>"
GEMMA_END_OF_TURN = "<end_of_turn>"
GEMMA_NL = "\n\n"


class GemmaPromptFormatter(PromptFormatter):
    NAME = "gemma"
    OUTPUT_ROLE = "assistant"
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


@registered_prompt_format_fn
def gemma(cuts: CutSet, tokenizer: TokenizerSpec):
    prompt = GemmaPromptFormatter(tokenizer)
    ans = defaultdict(list)
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut.first_non_padding_cut
        if cut.has_custom("context"):
            context = cut.context
        elif cut.has_custom("question"):
            context = cut.question
        else:
            context = cut.default_context

        turns = []
        if cut.has_custom("system_prompt"):
            turns.append({"role": "system_and_user", "slots": {"system": cut.system_prompt, "message": context}})
        else:
            turns.append({"role": "user", "slots": {"message": context}})
        if (answer := cut.supervisions[0].text) is not None:
            turns.append({"role": "assistant", "slots": {"message": answer}})

        for k, v in prompt.encode_dialog(turns).items():
            ans[k].append(v)

    return ans
