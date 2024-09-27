from collections import defaultdict

from lhotse import CutSet
from lhotse.cut import MixedCut

from nemo.collections.common.prompts import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec


class PlainPromptFormatter(PromptFormatter):
    """
    Plain prompt formatter: there is nothing being added to the user and assistants turns.
    """

    NAME = "plain"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn
def plain(cuts: CutSet, tokenizer: TokenizerSpec):
    prompt = PlainPromptFormatter(tokenizer)
    ans = defaultdict(list)
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut.first_non_padding_cut
        assert cut.has_custom("context"), f"Missing mandatory metadata key 'context' in {cut=}"

        turns = [{"role": "user", "slots": {"message": cut.context}}]
        if (answer := cut.supervisions[0].text) is not None:
            turns.append({"role": "assistant", "slots": {"message": answer}})

        for k, v in prompt.encode_dialog(turns).items():
            ans[k].append(v)

    return ans
