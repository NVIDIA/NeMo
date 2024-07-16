"""
Implemented following the guide at https://www.promptingguide.ai/models/gemma#gemma-7b-prompt-format
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

GEMMA_BOS = "<start_of_turn>"
GEMMA_END_OF_TURN = "<end_of_turn>"
GEMMA_NL = "\n\n"


class GemmaPromptFormatter(PromptFormatter):
    NAME = "gemma"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{GEMMA_BOS}user\n|message||audio_asr|{GEMMA_END_OF_TURN}\n{GEMMA_BOS}model\n",
            "slots": {
                "message": Modality.Text,
                "audio_speaker_identity": Modality.Audio("speaker_id"),
                "audio_asr": Modality.Audio("asr"),
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


"few-shot-1"
"few-shot-2"
"few-shot-3"

# |text||audio|
# |audio||text|
# |audio||text||audio||audio|
# <<<([audio|text]<sep>)+>>>
# "<sep>".join()


def usage():
    formatter = GemmaPromptFormatter(...)
    formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": Modality.Text, "audio_asr": torch.Tensor(...)}},
        ]
    )
