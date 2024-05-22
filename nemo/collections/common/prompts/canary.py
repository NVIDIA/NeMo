from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.collections.common.tokenizers.canary_tokenizer import CANARY_BOS, CANARY_EOS, CANARY_NOPNC, CANARY_PNC


class CanaryPromptFormatter(PromptFormatter):
    REGISTER_NAME = "canary"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{CANARY_BOS}|SOURCE_LANG||TASKNAME||TARGET_LANG||PNC|",
            "slots": {
                "|SOURCE_LANG|": str,
                "|TASKNAME|": str,
                "|TARGET_LANG|": str,
                "|PNC|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"|TEXT|{CANARY_EOS}",
            "slots": {
                "|TEXT|": str,
            },
        },
    }


def map_manifest_values_to_special_tokens(slot_values: dict[str, str]) -> dict[str, str]:
    slot_values = slot_values.copy()
    if not ((v := slot_values["|SOURCE_LANG|"]).startswith("<|") and v.endswith("|>")):
        slot_values["|SOURCE_LANG|"] = "<|" + slot_values["|SOURCE_LANG|"] + "|>"
    if not ((v := slot_values["|TARGET_LANG|"]).startswith("<|") and v.endswith("|>")):
        slot_values["|TARGET_LANG|"] = "<|" + slot_values["|TARGET_LANG|"] + "|>"
    if slot_values["|PNC|"] not in (CANARY_PNC, CANARY_NOPNC):
        slot_values["|PNC|"] = CANARY_PNC if slot_values["|PNC|"] in ("yes", "1", "True", "true") else CANARY_NOPNC
    if slot_values["|TASKNAME|"] not in ("<|transcribe|>", "<|translate|>"):
        slot_values["|TASKNAME|"] = "<|transcribe|>" if slot_values["|TASKNAME|"] == "asr" else "<|translate|>"
    return slot_values
