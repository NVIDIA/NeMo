from collections import defaultdict

from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.utils import ifnone

from nemo.collections.common.prompts import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import BOS_SLOT, EOS_SLOT, Modality, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec


class Llama2PromptFormatter(PromptFormatter):
    """
    This template has been validated to provide identical tokenized results to the official code
    in https://github.com/meta-llama/llama/blob/main/llama/generation.py
    """

    NAME = "llama2"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "system_and_user": {
            "template": f"{BOS_SLOT}[INST] <<SYS>>\n|system|\n<</SYS>>\n\n|message| [/INST]",
            "slots": {
                "system": Modality.Text,
                "message": Modality.Text,
            },
        },
        "user": {
            "template": "|bos|[INST] |message| [/INST]",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message| {EOS_SLOT}",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn
def llama2(cuts: CutSet, tokenizer: TokenizerSpec):
    prompt = Llama2PromptFormatter(tokenizer)
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


LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_HEADER_BEGIN = "<|start_header_id|>"
LLAMA3_HEADER_END = "<|end_header_id|>"
LLAMA3_END_OF_TURN = "<|eot_id|>"
LLAMA3_NL = "\n\n"


class Llama3PromptFormatter(PromptFormatter):
    """
    Implemented following the code at:
     https://github.com/meta-llama/llama3/blob/main/llama/test_tokenizer.py#L56
    """

    NAME = "llama3"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "preamble": {
            "template": LLAMA3_BOS,
        },
        "system": {
            "template": f"{LLAMA3_HEADER_BEGIN}system{LLAMA3_HEADER_END}{LLAMA3_NL}|message|{LLAMA3_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{LLAMA3_HEADER_BEGIN}user{LLAMA3_HEADER_END}{LLAMA3_NL}|message|{LLAMA3_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{LLAMA3_HEADER_BEGIN}assistant{LLAMA3_HEADER_END}{LLAMA3_NL}|message|{LLAMA3_END_OF_TURN}",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
