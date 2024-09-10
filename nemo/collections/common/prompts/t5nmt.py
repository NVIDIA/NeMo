import torch
from lhotse import CutSet, MonoCut
from lhotse.cut import MixedCut

from nemo.collections.common.prompts import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec


class T5NMTPromptFormatter(PromptFormatter):
    """
    The default prompt format for Megatron T5 based neural machine translation models.
    Based on: https://github.com/NVIDIA/NeMo/blob/ad5ef750e351edbb5eeb7eb6df2d0c804819600f/nemo/collections/nlp/models/machine_translation/megatron_nmt_model.py#L790
    """

    NAME = "t5nmt"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"|target_lang||message|",
            "slots": {
                "target_lang": Modality.Text,
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

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        # Automatically adds "<" and ">" to target lang token for T5 NMT.
        # Based on: https://github.com/NVIDIA/NeMo/blob/ad5ef750e351edbb5eeb7eb6df2d0c804819600f/nemo/collections/nlp/models/machine_translation/mt_enc_dec_model.py#L307
        if (val := slot_values.get("target_lang")) is not None:
            if not val.startswith("<") or not val.endswith(">"):
                slot_values["target_lang"] = f"<{val}>"
        return super().encode_turn(
            prompt_template=prompt_template, expected_slots=expected_slots, slot_values=slot_values
        )


@registered_prompt_format_fn
def t5nmt(cuts: CutSet, tokenizer: TokenizerSpec) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    formatter = T5NMTPromptFormatter(tokenizer)

    prompts_with_answers, prompts, answers = [], [], []
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut._first_non_padding_cut
        if not isinstance(cut, MonoCut):
            raise TypeError(
                f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
            )

        assert hasattr(cut, "context"), cut

        turns = [
            dict(
                role="user",
                # "message" slot is the audio portion of the cut; currently it is populated inside model's forward.
                slots={"target_lang": cut.context, "message": ""},
            ),
        ]
        if len(cut.supervisions) > 0 and cut.supervisions[0].text:
            turns.append(
                dict(
                    role="assistant",
                    slots={"message": cut.supervisions[0].text},
                )
            )
        encoded = formatter.encode_dialog(turns)
        prompts_with_answers.append(encoded["input_ids"])
        prompts.append(encoded["context_ids"])
        if "answer_ids" in encoded:
            answers.append(encoded["answer_ids"])
        else:
            answers.append([])

    return prompts_with_answers, prompts, answers
