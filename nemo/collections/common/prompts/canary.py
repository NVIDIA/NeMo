from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.collections.common.tokenizers.canary_tokenizer import CANARY_BOS, CANARY_EOS, CANARY_NOPNC, CANARY_PNC


class CanaryPromptFormatter(PromptFormatter):
    REGISTER_NAME = "canary"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{CANARY_BOS}|source_lang||task||target_lang||pnc|",
            "slots": {
                "source_lang": str,
                "task": str,
                "target_lang": str,
                "pnc": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|text|{CANARY_EOS}",
            "slots": {
                "text": str,
            },
        },
    }

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        # This method handles a level of indirection for Canary.
        # It maps values provided in trcfg to the actual special tokens
        # expected to be present in canary prompt.
        # It used to be done in prompt_format_fn isnide Dataset class corresponding to Canary,
        # but we are not using it here anymore.
        # This maps things such as '|task|: "asr"' to '|TASK|: "<|transcribe|>"'.
        slot_values = map_manifest_values_to_special_tokens(slot_values)
        return super().encode_turn(
            prompt_template=prompt_template, expected_slots=expected_slots, slot_values=slot_values
        )


def map_manifest_values_to_special_tokens(slot_values: dict[str, str]) -> dict[str, str]:
    slot_values = slot_values.copy()

    for k in ("source_lang", "target_lang"):
        if k in slot_values and not ((v := slot_values[k]).startswith("<|") and v.endswith("|>")):
            slot_values[k] = "<|" + slot_values[k] + "|>"

    k = "pnc"
    if k in slot_values and slot_values[k] not in (CANARY_PNC, CANARY_NOPNC):
        slot_values[k] = CANARY_PNC if slot_values[k] in ("yes", "1", "True", "true") else CANARY_NOPNC

    # Note: we re-map 'taskname' to 'task' for compatibility with earlier versions of Canary training.
    for k in ("task", "taskname"):
        if k in slot_values and slot_values[k] not in ("<|transcribe|>", "<|translate|>"):
            slot_values["task"] = "<|transcribe|>" if slot_values[k] == "asr" else "<|translate|>"

    return slot_values
